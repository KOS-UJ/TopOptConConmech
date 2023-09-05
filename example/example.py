from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import tri

from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import NonHomogenousSolver

from conmech.examples.p_slope_contact_law import make_slope_contact_law
from source.optimization import Optimization
from source.export_mesh import export_mesh, import_mesh, export_mesh_with_density


E = 10000
kappa = 0.4


@dataclass
class StaticSetup(StaticDisplacementProblem):
    grid_height: ... = 1.0
    elements_number: ... = (20, 40)
    mu_coef: ... = E / (1 + kappa)
    la_coef: ... = E * kappa / ((1 + kappa) * (1 - 2 * kappa))
    contact_law: ... = make_slope_contact_law(slope=0)

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([0, 0])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, -1]) if x[1] < 0.1 and x[0] > 1.9 else np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        return 0

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] < 0.5 and x[0] < 1,
        # dirichlet=lambda x: x[0] == 0 and x[1] > 0.5
        dirichlet=lambda x: x[0] == 0
    )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    setup = StaticSetup(mesh_type="cross")
    runner = NonHomogenousSolver(setup, "schur")

    optimizer = Optimization(
        setup=setup,
        simulation=runner,
        filter_radius=0.1,
        volume_fraction=0.3
    )
    density = optimizer.optimize(25)
    
    export_mesh_with_density(mesh=runner.body.mesh, mask=density > 0.8, filename="temp.msh")
    
    # mesh = import_mesh("temp.msh")
    # traingulation = tri.Triangulation(
    #     x=mesh.points[:, 0],
    #     y=mesh.points[:, 1],
    #     triangles=mesh.cells_dict['triangle']
    # )
    # plt.triplot(traingulation, color='#1f77b4')
    # plt.show()

if __name__ == "__main__":
    main(Config().init())
