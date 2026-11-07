"""Graph handedness synchronization from arbitrary-handed edge estimates."""

from typing import get_args

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common_lines.geometry import handedness
from common_lines.geometry.forward import compute_intrinsic_common_line_angles
from common_lines.graph_estimation import (
    HandednessMethod,
    ReconstitutionMethod,
    pairwise_orientation,
    select_consistent_handedness,
)
from common_lines.types import FloatArray


def _relative_graph(frames: FloatArray) -> FloatArray:
    """Use the image-i to image-j coordinate convention."""
    return frames[None, :] @ frames[:, None].swapaxes(-2, -1)


@pytest.mark.parametrize("handedness_method", get_args(HandednessMethod))
@pytest.mark.parametrize("method", get_args(ReconstitutionMethod))
def test_graph_recovers_known_frames_with_multiple_third_views(
    method: ReconstitutionMethod, handedness_method: HandednessMethod
) -> None:
    frames = Rotation.random(6, random_state=np.random.default_rng(12)).as_matrix()
    first, second = compute_intrinsic_common_line_angles(
        frames[:, None], frames[None, :]
    )
    # Self-pair angles are undefined and must never enter rotation conversion.
    np.fill_diagonal(first, np.inf)
    np.fill_diagonal(second, np.inf)
    with np.errstate(divide="raise", invalid="raise"):
        candidates, fallback = (
            pairwise_orientation.estimate_relative_orientation_candidates(
                first, second, n_third=3, random_seed=4, method=method
            )
        )
    assert not fallback.any()
    graph = select_consistent_handedness(
        candidates, reference_pair=(0, 1), method=handedness_method
    )
    truth = frames[None, :] @ frames[:, None].swapaxes(-1, -2)
    error = min(
        np.max(np.abs(graph - expected))
        for expected in (
            truth,
            handedness.mirror_relative_orientations_graph(truth),
        )
    )
    assert error < 1e-11


@pytest.mark.parametrize("method", get_args(ReconstitutionMethod))
def test_graph_handles_nearly_parallel_views(
    method: ReconstitutionMethod,
) -> None:
    frames = Rotation.from_euler(
        "ZYZ", [[0, 0, 0], [0.3, 0.7, -0.2], [0.3001, 1.2, 0.4], [-0.8, 0.5, 0.3]]
    ).as_matrix()
    first, second = compute_intrinsic_common_line_angles(
        frames[:, None], frames[None, :]
    )
    candidates, fallback = (
        pairwise_orientation.estimate_relative_orientation_candidates(
            first, second, n_third=3, random_seed=4, method=method
        )
    )
    assert not fallback.any()
    np.testing.assert_allclose(
        candidates @ candidates.swapaxes(-1, -2),
        np.broadcast_to(np.eye(3), candidates.shape),
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_allclose(np.linalg.det(candidates), 1.0, rtol=0, atol=2e-15)
    graph = select_consistent_handedness(
        candidates, reference_pair=(0, 1), method="star"
    )
    truth = frames[None, :] @ frames[:, None].swapaxes(-1, -2)
    assert (
        min(
            np.max(np.abs(graph - candidate))
            for candidate in (
                truth,
                handedness.mirror_relative_orientations_graph(truth),
            )
        )
        < 1e-8
    )


@pytest.mark.parametrize("handedness_method", get_args(HandednessMethod))
@pytest.mark.parametrize("reference_pair", [(0, 1), (4, 1)])
def test_select_consistent_handedness_recovers_noisy_independently_flipped_candidates(
    reference_pair: tuple[int, int],
    handedness_method: HandednessMethod,
) -> None:
    rng = np.random.default_rng(23)
    frames = Rotation.random(5, random_state=rng).as_matrix()
    truth = frames[None, :] @ frames[:, None].swapaxes(-1, -2)
    # Independent edge noise makes cycle consistency inexact. Synchronization
    # must preserve these measurements and only change their hands.
    perturbations = Rotation.from_rotvec(rng.normal(0, 0.01, (25, 3))).as_matrix()
    truth = perturbations.reshape(truth.shape) @ truth
    diagonal = np.arange(5)
    truth[diagonal, diagonal] = np.eye(3)
    mirrored = handedness.mirror_relative_orientations_graph(truth)
    flipped = rng.integers(0, 2, size=(5, 5), dtype=bool)
    candidates = np.where(flipped[..., None, None], mirrored, truth)
    original = candidates.copy()

    graph = select_consistent_handedness(
        candidates, reference_pair=reference_pair, method=handedness_method
    )

    # The measured reference fixes the global hand; all other
    # edge hands are independent and must not affect recovery.
    expected = mirrored if flipped[reference_pair] else truth
    np.testing.assert_allclose(graph, expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(candidates, original)
    np.testing.assert_allclose(
        select_consistent_handedness(
            graph, reference_pair=reference_pair, method=handedness_method
        ),
        graph,
        rtol=0,
        atol=1e-14,
    )


def test_orientation_graph_uses_i_to_j_coordinate_convention() -> None:
    """graph[i, j] transforms coordinates from image i to image j."""

    # Three hand-checkable frames. Each R_i maps coordinates from the
    # reference system into the coordinate system of image i.
    R0 = np.eye(3)
    R1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    R2 = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    frames = (R0, R1, R2)

    # Exact oriented common-line angles for these three frames.
    #
    # angle_1[i, j]: angle in image i
    # angle_2[i, j]: corresponding angle in image j
    angle_1 = np.array(
        [
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
            [0.0, 90.0, 0.0],
        ]
    )
    angle_2 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 90.0],
            [90.0, 90.0, 0.0],
        ]
    )

    candidates, used_fallback = (
        pairwise_orientation.estimate_relative_orientation_candidates(
            np.deg2rad(angle_1),
            np.deg2rad(angle_2),
            n_third=1,
            random_seed=0,
            method="voting",
        )
    )

    assert not np.any(used_fallback)

    graph = select_consistent_handedness(
        candidates,
        reference_pair=(0, 1),
        method="star",
    )

    for i in range(3):
        for j in range(3):
            expected = frames[j] @ frames[i].T
            assert np.allclose(graph[i, j], expected, atol=1.0e-12)


@pytest.mark.parametrize(
    "n_images, reference_pair, opposite_reference_hands",
    [(2, (0, 1), False), (2, (1, 0), True), (3, (0, 2), True), (6, (5, 0), False)],
)
def test_methods_agree_on_exact_independently_flipped_graphs(
    n_images: int, reference_pair: tuple[int, int], opposite_reference_hands: bool
) -> None:
    rng = np.random.default_rng(23)
    truth = _relative_graph(Rotation.random(n_images, random_state=rng).as_matrix())
    mirrored = handedness.mirror_relative_orientations_graph(truth)
    flipped = rng.integers(0, 2, (n_images, n_images), dtype=bool)
    # Exercise agreement and disagreement in the two reference directions. A lower-triangle
    # anchor must preserve its own measured hand, not the upper edge's hand.
    flipped[reference_pair[::-1]] = flipped[reference_pair] ^ opposite_reference_hands
    candidates = np.where(flipped[..., None, None], mirrored, truth)
    original = candidates.copy()
    expected = mirrored if flipped[reference_pair] else truth

    results = [
        select_consistent_handedness(
            candidates, reference_pair=reference_pair, method=method
        )
        for method in get_args(HandednessMethod)
    ]
    for result in results:
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(result[reference_pair], original[reference_pair])
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(candidates, original)


@pytest.mark.parametrize("method", get_args(HandednessMethod))
def test_in_plane_graphs_are_unchanged(
    method: HandednessMethod,
) -> None:
    frames = Rotation.from_euler("z", [[0.0], [0.3], [-1.2], [2.0]]).as_matrix()
    candidates = _relative_graph(frames)
    diagonal = np.arange(4)
    candidates[diagonal, diagonal] = np.eye(3)
    result = select_consistent_handedness(
        candidates, reference_pair=(3, 1), method=method
    )
    np.testing.assert_array_equal(result, candidates)


@pytest.mark.parametrize("method", get_args(HandednessMethod))
def test_noisy_selection_only_changes_edge_hands(method: HandednessMethod) -> None:
    # Large independent errors need not yield the same partition for both
    # methods. Neither method may replace measurements with predicted rotations.
    rng = np.random.default_rng(8)
    candidates = Rotation.random(25, random_state=rng).as_matrix().reshape(5, 5, 3, 3)
    original = candidates.copy()
    candidates.flags.writeable = False
    result = select_consistent_handedness(
        candidates, reference_pair=(4, 1), method=method
    )
    mirrored = handedness.mirror_relative_orientations_graph(original)
    edge_errors = np.minimum(
        np.max(np.abs(result - original), axis=(-2, -1)),
        np.max(np.abs(result - mirrored), axis=(-2, -1)),
    )
    np.testing.assert_array_equal(edge_errors[~np.eye(5, dtype=bool)], 0.0)
    np.testing.assert_array_equal(result[4, 1], original[4, 1])
    np.testing.assert_array_equal(
        result[np.arange(5), np.arange(5)], np.tile(np.eye(3), (5, 1, 1))
    )
    np.testing.assert_array_equal(candidates, original)
    assert not np.shares_memory(result, candidates)


@pytest.mark.parametrize(
    "shape, reference_pair",
    [((3, 3), (0, 1)), ((3, 3, 3, 3), (0, 0)), ((3, 3, 3, 3), (0, 3))],
)
def test_selector_rejects_invalid_graph_inputs(
    shape: tuple[int, ...], reference_pair: tuple[int, int]
) -> None:
    with pytest.raises(ValueError, match="async_graph|reference_pair"):
        select_consistent_handedness(
            np.zeros(shape), reference_pair=reference_pair, method="star"
        )


def test_selector_rejects_unknown_method() -> None:
    graph = np.broadcast_to(np.eye(3), (3, 3, 3, 3))
    with pytest.raises(ValueError, match="Unrecognized handedness selection method"):
        select_consistent_handedness(graph, reference_pair=(0, 1), method="unknown")  # type: ignore
