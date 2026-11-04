"""Mirror orientation graphs and select compatible handedness branches."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from common_lines.types import FloatArray

HANDEDNESS_REFLECTION = np.diag([1.0, 1.0, -1.0])


def mirror_relative_orientations_graph(graph: np.ndarray) -> np.ndarray:
    """
    Return the opposite-handed solution for a relative-orientation graph.

    Every relative rotation is conjugated with

        J = diag(1, 1, -1)

    so that

        mirrored[i, j] = J @ graph[i, j] @ J.

    The returned graph follows the same coordinate convention as the input:

        v_j = mirrored[i, j] @ v_i

    Parameters
    ----------
    graph
        Relative-orientation graph with shape
        ``(n_images, n_images, 3, 3)``.

    Returns
    -------
    numpy.ndarray
        Graph representing the globally mirrored handedness solution.
    """
    graph = np.asarray(graph, dtype=np.float64)
    if graph.ndim != 4 or graph.shape[2:] != (3, 3):
        raise ValueError(
            "graph must have shape (n_images, n_images, 3, 3). "
            f"Got array with shape {graph.shape}."
        )
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("The first two graph dimensions must be equal.")

    J = HANDEDNESS_REFLECTION
    return J @ graph @ J


def mirror_transformation_matrices(
    rotations: Iterable[np.ndarray],
) -> tuple[FloatArray, ...]:
    """Return opposite-handed frames via J @ R @ J, J = diag(1, 1, -1).

    Matrices may have arbitrary leading batch dimensions. Each matrix maps
    coordinates from the common reference frame to its image frame.
    """
    J = HANDEDNESS_REFLECTION
    return tuple(J @ rotation @ J for rotation in rotations)
