import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri

from conmech.mesh.mesh import Mesh


def plot_density(mesh: Mesh, density: np.ndarray, ratio: float, file_name: str):
    triangulation = tri.Triangulation(
        x=mesh.nodes[:, 0], y=mesh.nodes[:, 1], triangles=mesh.elements
    )
    plt.tripcolor(triangulation, density, cmap="Greys", vmin=0, vmax=1)

    ax = plt.gca()
    if ratio is not None:
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.colorbar()
    plt.grid()
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()


def plot_displacements(
    mesh: Mesh,
    displacements: np.ndarray,
    density: np.ndarray,
    scale_factor: float,
    ratio: float,
    file_name: str,
):
    before = tri.Triangulation(x=mesh.nodes[:, 0], y=mesh.nodes[:, 1], triangles=mesh.elements)
    before.set_mask(density < 0.08)
    plt.triplot(before, color="#1f77b4")
    after = tri.Triangulation(
        x=mesh.nodes[:, 0] + displacements[:, 0] * scale_factor,
        y=mesh.nodes[:, 1] + displacements[:, 1] * scale_factor,
        triangles=mesh.elements,
    )
    after.set_mask(density < 0.08)
    plt.triplot(after, color="#ff7f0e")

    ax = plt.gca()
    if ratio is not None:
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.grid()
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
