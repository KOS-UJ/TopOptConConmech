import numpy as np


def area_of_triangle(nodes: np.ndarray) -> float:
    # coords = np.array([['x1', 'y1'], ['x2', 'y2'], ['x3', 'y3']])
    double_t = nodes[1:3].T - np.expand_dims(nodes[0], 1)
    area_t = np.abs(np.linalg.det(double_t)) / 2
    return area_t


def center_of_mass(nodes: np.ndarray) -> np.ndarray:
    return np.sum(nodes, 0) / nodes.shape[0]
