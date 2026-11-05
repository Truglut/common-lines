"""Select compatible handedness branches of measured relative orientations."""

import itertools
from typing import Literal

import numpy as np

from common_lines.geometry.handedness import HANDEDNESS_REFLECTION

from .j_synchronization import _synchronize_with_j_sync

HandednessMethod = Literal["star", "synchronization"]


def select_consistent_handedness(
    async_graph: np.ndarray,
    *,
    reference_pair: tuple[int, int],
    method: HandednessMethod,
) -> np.ndarray:
    """
    Select one graph-wide handedness branch while preserving edge measurements.

    Each measured rotation has two possible hands, ``R`` and ``J @ R @ J``,
    where ``J = diag(1, 1, -1)``. Select between them using compatibility across
    image pairs. The reference edge fixes the arbitrary global hand; the data
    alone cannot distinguish it from the globally mirrored solution.

    Parameters
    ----------
    async_graph : np.ndarray
        Array with shape ``(n_images, n_images, 3, 3)``, with at least two images.
        Each ``async_graph[i, j]`` maps coordinates from image ``i`` to image
        ``j``, up to handedness. Edges may have independent hands and measurement
        errors. Rotation matrices are assumed; orthogonality and determinants
        are not checked.
    reference_pair : tuple[int, int]
        Ordered pair of distinct, in-range integer image indices. The measured
        rotation ``async_graph[reference_pair]`` is retained in the output,
        including when the pair refers to a lower-triangle entry.
    method : HandednessMethod
        ``"star"`` selects a reference star and aligns the remaining edge hands
        with it. ``"synchronization"`` combines triplet constraints through a
        spectral J-synchronization step.

    Returns
    -------
    numpy.ndarray
        New float64 array with the same shape and coordinate convention as the
        input. Each off-diagonal entry is its measured rotation or its mirrored
        branch; diagonal entries are identity matrices. The input is unchanged.

    Raises
    ------
    ValueError
        If the graph shape, reference pair, or method is invalid.

    Notes
    -----
    Handedness selection does not enforce exact inverse or cycle consistency
    of noisy rotations. The two methods can choose different branches on noisy
    data. Obtain the opposite global solution with
    ``common_lines.geometry.handedness.mirror_relative_orientations_graph``.
    """
    async_graph = np.asarray(async_graph, dtype=np.float64)
    if (
        async_graph.ndim != 4
        or async_graph.shape[-2:] != (3, 3)
        or async_graph.shape[0] != async_graph.shape[1]
    ):
        raise ValueError(
            "async_graph must be an array of shape (n_images, n_images, 3, 3)"
        )
    n_images = async_graph.shape[0]
    if (
        not isinstance(reference_pair, tuple)
        or len(reference_pair) != 2
        or not all(
            isinstance(index, (int, np.integer))
            and not isinstance(index, (bool, np.bool_))
            and 0 <= index < n_images
            for index in reference_pair
        )
        or reference_pair[0] == reference_pair[1]
    ):
        raise ValueError(
            "reference_pair must contain two distinct integer image indices "
            f"in [0, {n_images})."
        )

    if method == "star":
        return _synchronize_with_star(async_graph, reference_pair=reference_pair)
    elif method == "synchronization":
        return _synchronize_with_j_sync(async_graph, reference_pair=reference_pair)

    raise ValueError(f"Unrecognized handedness selection method: {method}")


def _synchronize_with_star(
    async_graph: np.ndarray, *, reference_pair: tuple[int, int]
) -> np.ndarray:
    """
    Select one globally consistent handedness branch for each graph edge.

    The measured rotation at ``reference_pair`` fixes the arbitrary global hand. A
    reference star is then selected using triangles that contain both reference
    images. Every remaining directed edge keeps its independent estimate and
    only chooses the branch closest to the rotation predicted by that star.

    Inputs are normalized and validated by ``select_consistent_handedness``.
    Return a new float64 graph with identity diagonal entries; ties retain the
    measured branch. No measured rotation is replaced by a star prediction.
    """
    n_images = async_graph.shape[0]
    reference_i, reference_j = reference_pair

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
