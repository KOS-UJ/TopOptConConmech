from dataclasses import dataclass

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import NonHomogenousSolver

from conmech.examples.p_slope_contact_law import make_slope_contact_law
from source.optimization import Optimization


@dataclass
class StaticSetup(StaticDisplacementProblem):
    grid_height: ... = 1.0
    elements_number: ... = (2, 5)
    mu_coef: ... = 4
    la_coef: ... = 4
    contact_law: ... = make_slope_contact_law(slope=1)

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([0, 0])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, 0.1]) if x[1] < 0.2 and x[0] > 2.2 else np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        return 0

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0 and x[0] < 0.2, 
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
        simulation=runner
    )
    optimizer.optimize(10)


if __name__ == "__main__":
    main(Config().init())
