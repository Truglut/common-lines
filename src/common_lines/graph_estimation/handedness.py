"""Mirror orientation graphs and select compatible handedness branches."""

from __future__ import annotations

import itertools

import numpy as np

from common_lines.reconstitution.triplets import HANDEDNESS_REFLECTION


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


def select_consistent_handedness(
    async_graph: np.ndarray, *, reference_pair: tuple[int, int]
) -> np.ndarray:
    """
    Select one globally consistent handedness branch for each graph edge.

    The measured rotation at ``reference_pair`` fixes the arbitrary global hand. A
    reference star is then selected using triangles that contain both reference
    images. Every remaining directed edge keeps its independent estimate and
    only chooses the branch closest to the rotation predicted by that star.
    """
    async_graph = np.asarray(async_graph, dtype=np.float64)
    if async_graph.ndim != 4 or async_graph.shape[2:] != (3, 3):
        raise ValueError(
            "async_graph must have shape (n_images, n_images, 3, 3). "
            f"Got array with shape {async_graph.shape}."
        )
    if async_graph.shape[0] != async_graph.shape[1]:
        raise ValueError("The first two dimensions in async_graph must be equal.")

    n_images = async_graph.shape[0]
    reference_i, reference_j = reference_pair
    if not (
        0 <= reference_i < n_images
        and 0 <= reference_j < n_images
        and reference_i != reference_j
    ):
        raise ValueError(
            "reference_pair must contain two different valid image indices."
        )

    graph = np.empty((n_images, n_images, 3, 3), dtype=np.float64)
    diagonal = np.arange(n_images)
    graph[diagonal, diagonal] = np.eye(3)

    # The measured reference edge fixes one of the two global hands.
    graph[reference_i, reference_j] = async_graph[reference_i, reference_j]

    # Select a handedness-consistent star around the first reference image.
    for k in range(n_images):
        if k == reference_i or k == reference_j:
            continue

        measured_ik = async_graph[reference_i, k]
        mirrored_ik = HANDEDNESS_REFLECTION @ measured_ik @ HANDEDNESS_REFLECTION

        measured_jk = async_graph[reference_j, k]
        mirrored_jk = HANDEDNESS_REFLECTION @ measured_jk @ HANDEDNESS_REFLECTION

        min_error_measured = min(
            np.linalg.norm(
                measured_jk @ graph[reference_i, reference_j] - measured_ik, ord="fro"
            ),
            np.linalg.norm(
                mirrored_jk @ graph[reference_i, reference_j] - measured_ik, ord="fro"
            ),
        )

        min_error_mirrored = min(
            np.linalg.norm(
                measured_jk @ graph[reference_i, reference_j] - mirrored_ik, ord="fro"
            ),
            np.linalg.norm(
                mirrored_jk @ graph[reference_i, reference_j] - mirrored_ik, ord="fro"
            ),
        )

        if min_error_mirrored < min_error_measured:
            graph[reference_i, k] = mirrored_ik
        else:
            graph[reference_i, k] = measured_ik

    # Choose every other directed edge using the selected reference star.
    for i, j in itertools.permutations(range(n_images), 2):
        if i == reference_i:
            continue

        # R_{i->j} = R_{ref->j} @ R_{ref->i}^{-1}.
        # Rotation matrices are orthogonal, so the inverse is the transpose.
        predicted = graph[reference_i, j] @ graph[reference_i, i].T

        measured = async_graph[i, j]
        mirrored = HANDEDNESS_REFLECTION @ measured @ HANDEDNESS_REFLECTION

        direct_distance = np.linalg.norm(measured - predicted, ord="fro")
        mirrored_distance = np.linalg.norm(mirrored - predicted, ord="fro")

        graph[i, j] = mirrored if mirrored_distance < direct_distance else measured

    return graph
