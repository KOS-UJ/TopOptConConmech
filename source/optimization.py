import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri

from conmech.simulations.problem_solver import NonHomogenousSolver
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.mesh.mesh import Mesh


class Optimization:

    def __init__(
            self,
            setup: StaticDisplacementProblem,
            simulation: NonHomogenousSolver,
            penalty: float = 3,
            volume_fraction: float = 0.6,
            filter_radius: float = 1.
    ):
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


    def bisection(self, x: np.ndarray, comp_deriv: np.ndarray, num_dumping: float = 0.5) -> np.ndarray:
        step = 0.2
        lower = 1e-9
        upper = 1e9

        lower_limit = np.maximum(1e-4, x - step)
        upper_limit = np.minimum(1., x + step)

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
        centers = np.array([center_of_mass(self.mesh.initial_nodes[el_nodes]) for el_nodes in self.mesh.elements])
        diffs = centers[:, None] - centers
        distances = np.linalg.norm(diffs, axis=2)
        elem_filter_weights = (self.filter_radius - distances).clip(min=0)
        return elem_filter_weights

    def get_elems_volumes(self):
        volumes = np.array([
            area_of_triangle(self.mesh.initial_nodes[nodes_ids])
            for nodes_ids in self.mesh.elements
        ])
        return volumes

    def compute_elems_compliance(self, displacement: np.ndarray, elem_stiff: np.ndarray):
        # elements_compliance = np.empty_like(density)
        # for elem_idx in range(elements_compliance.size):
        #     elem_displacement = np.expand_dims(displacement[self.base_func_ids[elem_idx]], 1)
        #     elements_compliance[elem_idx] = elem_displacement.T @ elem_stiff[elem_idx] @ elem_displacement
        # 1335 ms

        elements_compliance = np.squeeze(np.array([
            displacement[None, base_funcs] @ elem_stiff_mat @ displacement[base_funcs, None]
            for base_funcs, elem_stiff_mat in zip(self.base_func_ids, elem_stiff)
        ]))
        # 861 ms
        return elements_compliance

    def create_local_stifmats(self):
        w_mat = self.simulation.body._local_stifness_matrices
        mu = self.setup.mu_coef
        lambda_ = self.setup.la_coef

        A_11 = (2 * mu + lambda_) * w_mat[0] + mu * w_mat[3]
        A_12 = mu * w_mat[2] + lambda_ * w_mat[1]
        A_21 = lambda_ * w_mat[2] + mu * w_mat[1]
        A_22 = mu * w_mat[0] + (2 * mu + lambda_) * w_mat[3]
        return np.block([[A_11, A_12], [A_21, A_22]])

    def optimize(self, iteration_limit: int = 100) -> np.ndarray:

        density = np.full(self.mesh.elements.shape[0], fill_value=self.volume_fraction)

        iteration = 0
        change = 1.

        elem_stiff = self.create_local_stifmats()

        while change > 1e-2 and iteration < iteration_limit:
            iteration += 1

            self.simulation.update_density(density=density ** self.penalty)
            result = self.simulation.solve(initial_displacement=self.setup.initial_displacement)
            displacement = np.hstack((result.displacement[:, 0], result.displacement[:, 1]))

            elements_compliance = self.compute_elems_compliance(
                displacement=displacement,
                elem_stiff=elem_stiff
            )

            compliance = np.sum((density ** self.penalty) * elements_compliance)
            comp_derivative = -self.penalty * (density ** (self.penalty - 1)) * elements_compliance
            print(f'iteration: {iteration}')
            print(f'compliance = {compliance}')

            comp_derivative = self.mesh_independency_filter(
                comp_deriv=comp_derivative,
                density=density
            )

            old_density = density.copy()
            density = self.bisection(x=density, comp_deriv=comp_derivative)
            print(f'volume = {np.sum(density * self.elem_volumes)}')
            change = np.max(np.abs(density - old_density))
            print(f'change = {change}')

            axes_sizes = self.setup.elements_number
            plot_displacements(
                mesh=self.mesh,
                displacements=result.displacement,
                density=density,
                scale_factor=1,
                ratio=axes_sizes[0] / axes_sizes[1],
                file_name=f'output/displacement{iteration}'
            )
            plot_density(
                mesh=self.mesh,
                density=density,
                ratio=axes_sizes[0] / axes_sizes[1],
                file_name=f'output/density{iteration}'
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


def plot_density(mesh: Mesh, density: np.ndarray, ratio: float, file_name: str):
    triangulation = tri.Triangulation(
        x=mesh.initial_nodes[:, 0],
        y=mesh.initial_nodes[:, 1],
        triangles=mesh.elements
    )
    plt.tripcolor(triangulation, density, cmap='Greys', vmin=0, vmax=1)
    
    ax = plt.gca()
    if ratio is not None:
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
    
    plt.colorbar()
    plt.grid()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def plot_displacements(mesh: Mesh, displacements: np.ndarray, density: np.ndarray, scale_factor: float, ratio: float, file_name: str):

    before = tri.Triangulation(
        x=mesh.initial_nodes[:, 0],
        y=mesh.initial_nodes[:, 1],
        triangles=mesh.elements
    )
    before.set_mask(density < 0.08)
    plt.triplot(before, color='#1f77b4')
    after = tri.Triangulation(
        x=mesh.initial_nodes[:, 0] + displacements[:, 0] * scale_factor,
        y=mesh.initial_nodes[:, 1] + displacements[:, 1] * scale_factor,
        triangles=mesh.elements
    )
    after.set_mask(density < 0.08)
    plt.triplot(after, color='#ff7f0e')

    ax = plt.gca()
    if ratio is not None:
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.grid()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def area_of_triangle(nodes: np.ndarray) -> float:
    # coords = np.array([['x1', 'y1'], ['x2', 'y2'], ['x3', 'y3']])
    double_t = nodes[1:3].T - np.expand_dims(nodes[0], 1)
    area_t = np.abs(np.linalg.det(double_t)) / 2
    return area_t


def center_of_mass(nodes: np.ndarray) -> np.ndarray:
    return np.sum(nodes, 0) / nodes.shape[0]
