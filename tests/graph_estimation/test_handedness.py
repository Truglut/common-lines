"""Graph handedness selection from reconstructed pair candidates."""

from typing import get_args

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common_lines.geometry.forward import compute_intrinsic_common_line_angles
from common_lines.graph_estimation import (
    ReconstitutionMethod,
    handedness,
    pairwise_orientation,
)
from common_lines.reconstitution.matrix import get_transformation_for_second_image
from common_lines.reconstitution.triplets import CommonLineAngles, CommonLineTriplet


@pytest.mark.parametrize("method", ["averaging", "voting"])
def test_regular_graph_matches_matrix_solver_up_to_global_handedness(
    method: ReconstitutionMethod,
) -> None:
    rng = np.random.default_rng(12)
    frames = Rotation.random(3, random_state=rng).as_matrix()
    first, second = compute_intrinsic_common_line_angles(
        frames[:, None], frames[None, :]
    )
    noisy_first = first + rng.normal(0, 0.04, first.shape)
    noisy_second = second + rng.normal(0, 0.04, second.shape)

    for angle_1, angle_2 in [(first, second), (noisy_first, noisy_second)]:
        actual, fallback = (
            pairwise_orientation.estimate_relative_orientation_candidates(
                angle_1, angle_2, n_third=1, random_seed=4, method=method
            )
        )
        # Three images give one triplet per edge, so the public matrix solver
        # provides an independent oracle without reproducing graph averaging.
        assert not fallback.any()
        expected = np.empty_like(actual)
        for i in range(3):
            expected[i, i] = np.eye(3)
            for j in range(3):
                if i == j:
                    continue
                k = 3 - i - j
                triplet = CommonLineTriplet(
                    CommonLineAngles(angle_1[i, j], angle_2[i, j]),
                    CommonLineAngles(angle_1[i, k], angle_2[i, k]),
                    CommonLineAngles(angle_1[j, k], angle_2[j, k]),
                )
                expected[i, j] = get_transformation_for_second_image(triplet)
        # Candidate order can differ independently on each directed edge.
        edge_errors = np.minimum(
            np.max(np.abs(actual - expected), axis=(-3, -2, -1)),
            np.max(np.abs(actual - expected[:, :, ::-1]), axis=(-3, -2, -1)),
        )
        np.testing.assert_allclose(edge_errors, 0.0, rtol=0, atol=1e-11)
        actual_graph = handedness.select_consistent_handedness(
            actual, reference_pair=(0, 1)
        )
        expected_graph = handedness.select_consistent_handedness(
            expected, reference_pair=(0, 1)
        )
        assert (
            min(
                np.max(np.abs(actual_graph - candidate))
                for candidate in (
                    expected_graph,
                    handedness.mirror_relative_orientations_graph(expected_graph),
                )
            )
            < 1e-11
        )


@pytest.mark.parametrize("method", get_args(ReconstitutionMethod))
@pytest.mark.parametrize("n_third", [1, 3])
def test_graph_recovers_known_frames_with_multiple_third_views(
    method: ReconstitutionMethod, n_third: int
) -> None:
    frames = Rotation.random(6, random_state=np.random.default_rng(12)).as_matrix()
    first, second = compute_intrinsic_common_line_angles(
        frames[:, None], frames[None, :]
    )
    candidates, fallback = (
        pairwise_orientation.estimate_relative_orientation_candidates(
            first, second, n_third=n_third, random_seed=4, method=method
        )
    )
    assert not fallback.any()
    graph = handedness.select_consistent_handedness(candidates, reference_pair=(0, 1))
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
def test_graph_handles_nearly_parallel_and_degenerate_views(
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
    graph = handedness.select_consistent_handedness(candidates, reference_pair=(0, 1))
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

    candidates, fallback = (
        pairwise_orientation.estimate_relative_orientation_candidates(
            np.zeros((3, 3)), np.zeros((3, 3)), n_third=3, random_seed=4, method=method
        )
    )
    np.testing.assert_array_equal(fallback, ~np.eye(3, dtype=bool))
    np.testing.assert_allclose(candidates, np.broadcast_to(np.eye(3), candidates.shape))


@pytest.mark.parametrize("reference_pair", [(0, 1), (2, 4), (4, 1)])
def test_select_consistent_handedness_recovers_independently_flipped_candidates(
    reference_pair: tuple[int, int],
) -> None:
    rng = np.random.default_rng(23)
    frames = Rotation.random(5, random_state=rng).as_matrix()
    truth = frames[None, :] @ frames[:, None].swapaxes(-1, -2)
    mirrored = handedness.mirror_relative_orientations_graph(truth)
    candidates = np.stack((truth, mirrored), axis=2)
    flipped = rng.integers(0, 2, size=(5, 5), dtype=bool)
    candidates = np.where(
        flipped[..., None, None, None], candidates[:, :, ::-1], candidates
    )
    original = candidates.copy()

    graph = handedness.select_consistent_handedness(
        candidates, reference_pair=reference_pair
    )

    # The reference's first candidate fixes the global hand; all other
    # candidate orderings are independent and must not affect recovery.
    expected = mirrored if flipped[reference_pair] else truth
    np.testing.assert_allclose(graph, expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(candidates, original)


@pytest.mark.parametrize("method", get_args(ReconstitutionMethod))
def test_orientation_graph_uses_i_to_j_coordinate_convention(
    method: ReconstitutionMethod,
):
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
            method=method,
        )
    )

    assert not np.any(used_fallback)

    graph = handedness.select_consistent_handedness(
        candidates,
        reference_pair=(0, 1),
    )

    for i in range(3):
        for j in range(3):
            expected = frames[j] @ frames[i].T
            assert np.allclose(graph[i, j], expected, atol=1.0e-12)
