import numpy as np

from conmech.simulations.problem_solver import NonHomogenousSolver
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.dynamics.factory.dynamics_factory_method import get_factory

from source.plotting import plot_density, plot_displacements
from source.mesh_utils import center_of_mass, area_of_triangle


class Optimization:
    def __init__(
        self,
        setup: StaticDisplacementProblem,
        simulation: NonHomogenousSolver,
        penalty: float = 3,
        volume_fraction: float = 0.6,
        filter_radius: float = 1.0,
    ):
        self.dimension = 2
        self.setup = setup
        self.simulation = simulation
        self.mesh = simulation.body.mesh

        self.penalty = penalty
        self.volume_fraction = volume_fraction
        self.filter_radius = filter_radius

        self.elem_volumes = self.get_elems_volumes()
        self.volume = np.sum(self.elem_volumes)

        self.base_func_ids = [
            np.hstack((nodes_ids, nodes_ids + self.mesh.nodes_count))
            for nodes_ids in self.mesh.elements
        ]
        self.elem_filter_weights = self.get_elements_surrounding()
        # self.plots_utils = PlotsUtils(
        #     mesh=self.mesh,
        #     penalty=self.penalty,
        #     elem_volumes=self.elem_volumes
        # )

    def bisection(
        self, x: np.ndarray, comp_deriv: np.ndarray, num_dumping: float = 0.5
    ) -> np.ndarray:
        step = 0.2
        lower = 1e-9
        upper = 1e9

        lower_limit = np.maximum(1e-4, x - step)
        upper_limit = np.minimum(1.0, x + step)

        x_new: np.ndarray
        while upper - lower > 1e-9:
            mid = lower + (upper - lower) / 2

            # B_e = -(compliance derivative / (lambda * volume derivative))
            beta = (-comp_deriv / (mid * self.elem_volumes)) ** num_dumping
            x_new = np.clip(beta * x, lower_limit, upper_limit)

            # volume [np.sum(self.elem_volumes * x_new)] is monotonously decreasing function of lagrange multiplayer [mid]
            if np.sum(self.elem_volumes * x_new) < self.volume_fraction * self.volume:
                upper = mid
            else:
                lower = mid
        return x_new

    def mesh_independency_filter(self, comp_deriv: np.ndarray, density: np.ndarray):
        neighbours_influence = np.sum((density * comp_deriv) * self.elem_filter_weights, axis=1)
        inertia = density * np.sum(self.elem_filter_weights, axis=1)
        new_comp_deriv = neighbours_influence / inertia
        return new_comp_deriv

    def get_elements_surrounding(self):
        centers = np.array(
            [center_of_mass(self.mesh.nodes[el_nodes]) for el_nodes in self.mesh.elements]
        )
        diffs = centers[:, None] - centers
        distances = np.linalg.norm(diffs, axis=2)
        elem_filter_weights = (self.filter_radius - distances).clip(min=0)
        return elem_filter_weights

    def get_elems_volumes(self):
        volumes = np.array(
            [area_of_triangle(self.mesh.nodes[nodes_ids]) for nodes_ids in self.mesh.elements]
        )
        return volumes

    def compute_elems_compliance(self, displacement: np.ndarray, elem_stiff: np.ndarray):
        # elements_compliance = np.empty_like(density)
        # for elem_idx in range(elements_compliance.size):
        #     elem_displacement = np.expand_dims(displacement[self.base_func_ids[elem_idx]], 1)
        #     elements_compliance[elem_idx] = elem_displacement.T @ elem_stiff[elem_idx] @ elem_displacement
        # 1335 ms

        elements_compliance = np.squeeze(
            np.array(
                [
                    displacement[None, base_funcs] @ elem_stiff_mat @ displacement[base_funcs, None]
                    for base_funcs, elem_stiff_mat in zip(self.base_func_ids, elem_stiff)
                ]
            )
        )
        # 861 ms
        return elements_compliance

    def create_local_stifmats(self):
        w_mat = self.simulation.body.dynamics._local_stifness_matrices
        mu = self.setup.mu_coef
        lambda_ = self.setup.la_coef
        factory = get_factory(self.dimension)
        return factory.calculate_constitutive_matrices(W=w_mat, mu=mu, lambda_=lambda_)

    def optimize(self, iteration_limit: int = 100) -> np.ndarray:
        density = np.full(self.mesh.elements.shape[0], fill_value=self.volume_fraction)

        iteration = 0
        change = 1.0

        elem_stiff = self.create_local_stifmats()

        while change > 1e-2 and iteration < iteration_limit:
            iteration += 1

            self.simulation.update_density(density=density**self.penalty)
            result = self.simulation.solve(initial_displacement=self.setup.initial_displacement)
            displacement = np.hstack((result.displacement[:, 0], result.displacement[:, 1]))

            elements_compliance = self.compute_elems_compliance(
                displacement=displacement, elem_stiff=elem_stiff
            )

            compliance = np.sum((density**self.penalty) * elements_compliance)
            comp_derivative = -self.penalty * (density ** (self.penalty - 1)) * elements_compliance
            print(f"iteration: {iteration}")
            print(f"compliance = {compliance}")

            comp_derivative = self.mesh_independency_filter(
                comp_deriv=comp_derivative, density=density
            )

            old_density = density.copy()
            density = self.bisection(x=density, comp_deriv=comp_derivative)
            print(f"volume = {np.sum(density * self.elem_volumes)}")
            change = np.max(np.abs(density - old_density))
            print(f"change = {change}")

            axes_sizes = self.mesh.scale
            plot_displacements(
                mesh=self.mesh,
                displacements=result.displacement,
                density=density,
                scale_factor=1,
                ratio=axes_sizes[0] / axes_sizes[1],
                file_name=f"output/displacement{iteration}",
            )
            plot_density(
                mesh=self.mesh,
                density=density,
                ratio=axes_sizes[0] / axes_sizes[1],
                file_name=f"output/density{iteration}",
            )

        #     if iteration < 25 or iteration % 5 == 0 or iteration in [32, 64]:
        #         self.plots_utils.make_plots(
        #             displacement=displacement,
        #             density=density,
        #             comp_derivative=comp_derivative,
        #             elements_compliance=elements_compliance,
        #             iteration=iteration
        #         )
        # self.plots_utils.draw_final_design(density)
        return density
