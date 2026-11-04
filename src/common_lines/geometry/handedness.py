"""Reflect frames and relative-orientation graphs into the opposite global hand."""

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
        New float64 array of the same shape representing the globally mirrored
        handedness solution. The input is unchanged; applying this operation
        twice recovers its original values.

    Raises
    ------
    ValueError
        If the graph does not have shape ``(n_images, n_images, 3, 3)``.
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
    """
    Return opposite-handed frames via ``J @ R @ J``, ``J = diag(1, 1, -1)``.

    Parameters
    ----------
    rotations : Iterable[numpy.ndarray]
        Frame matrices mapping coordinates from a common reference system to
        image coordinates. Each element has shape ``(..., 3, 3)`` and may carry
        arbitrary leading batch dimensions. A stack with shape ``(n, 3, 3)``
        is also accepted as an iterable of ``n`` frames.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        New arrays in input order, preserving each element's shape and frame
        convention. The input is unchanged. An empty iterable gives an empty
        tuple. Applying the operation twice recovers the original values.

    Notes
    -----
    This changes the hand of all supplied frames together. It does not select
    consistent hands for independently ambiguous relative-orientation edges.
    """
    J = HANDEDNESS_REFLECTION
    return tuple(J @ rotation @ J for rotation in rotations)
