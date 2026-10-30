"""Mirror orientation graphs and select compatible handedness branches."""

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
    candidates: np.ndarray, *, reference_pair: tuple[int, int]
) -> np.ndarray:
    """
    Select one globally consistent handedness branch for each graph edge.

    The first branch of ``reference_pair`` fixes the arbitrary global hand. A
    reference star is then selected using triangles that contain both reference
    images. Every remaining directed edge keeps its independent estimate and
    only chooses the branch closest to the rotation predicted by that star.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 5 or candidates.shape[2:] != (2, 3, 3):
        raise ValueError(
            "candidates must have shape (n_images, n_images, 2, 3, 3). "
            f"Got array with shape {candidates.shape}."
        )
    if candidates.shape[0] != candidates.shape[1]:
        raise ValueError("The first two candidate dimensions must be equal.")

    n_images = candidates.shape[0]
    reference_i, reference_j = reference_pair

    graph = np.empty((n_images, n_images, 3, 3), dtype=np.float64)
    diagonal = np.arange(n_images)
    graph[diagonal, diagonal] = np.eye(3)

    # The first candidate of the reference edge fixes one of the two global hands.
    graph[reference_i, reference_j] = candidates[reference_i, reference_j, 0]

    # Select a handedness-consistent star around the first reference image.
    for k in range(n_images):
        if k == reference_i or k == reference_j:
            continue

        best_error = np.inf
        best_reference_branch = 0

        for branch_ik in range(2):
            for branch_jk in range(2):
                # With graph[a, b] mapping a -> b, the path
                # reference_i -> reference_j -> k is
                # R_{j->k} @ R_{i->j}.
                predicted = (
                    candidates[reference_j, k, branch_jk]
                    @ graph[reference_i, reference_j]
                )
                error = np.linalg.norm(
                    predicted - candidates[reference_i, k, branch_ik],
                    ord="fro",
                )

                if error < best_error:
                    best_error = error
                    best_reference_branch = branch_ik

        graph[reference_i, k] = candidates[
            reference_i,
            k,
            best_reference_branch,
        ]

    # Choose every other directed edge using the selected reference star.
    for i, j in itertools.permutations(range(n_images), 2):
        if i == reference_i:
            continue

        # R_{i->j} = R_{ref->j} @ R_{ref->i}^{-1}.
        # Rotation matrices are orthogonal, so the inverse is the transpose.
        predicted = graph[reference_i, j] @ graph[reference_i, i].T

        direct_distance = np.linalg.norm(
            candidates[i, j, 0] - predicted,
            ord="fro",
        )
        mirrored_distance = np.linalg.norm(
            candidates[i, j, 1] - predicted,
            ord="fro",
        )

        branch = 0 if direct_distance <= mirrored_distance else 1
        graph[i, j] = candidates[i, j, branch]

    return graph
