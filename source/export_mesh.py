import meshio
import numpy as np

from conmech.mesh.mesh import Mesh


def export_mesh(mesh: Mesh, filename: str):
    meshio.write_points_cells(
        points=mesh.initial_nodes,
        cells={
            "triangle": mesh.elements
        },
        filename=filename
    )


def import_mesh(filename: str):
    mesh = meshio.read(filename)
    # print(mesh.points)
    # print(mesh.cells_dict['triangle'])
    # TODO create conmech mesh object
    return mesh


def export_mesh_with_density(mesh: Mesh, mask: np.ndarray, filename: str):
    is_node_removed = np.full(mesh.nodes_count, fill_value=True, dtype=bool)
    for element_index, element_nodes in enumerate(mesh.elements):
        if not mask[element_index]:
            continue
        is_node_removed[element_nodes] = False
    removed_nodes = np.cumsum(is_node_removed)

    elements = [node_idx - removed_nodes[node_idx] for node_idx in mesh.elements[mask]]
    nodes = mesh.initial_nodes[np.logical_not(is_node_removed)]

    meshio.write_points_cells(
        points=nodes,
        cells={
            "triangle": elements
        },
        filename=filename
    )
