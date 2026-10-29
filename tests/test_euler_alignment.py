"""Geometric correctness and regular-case comparisons with matrix alignment."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common_lines import graph_estimation
from common_lines.euler import euler_zyz_to_matrix
from common_lines.forward import compute_intrinsic_common_line_angles
from common_lines.reconstitution.euler import alignment_euler_angles
from common_lines.reconstitution.matrix import (
    get_transformation_for_second_image,
)
from common_lines.reconstitution.triplets import (
    HANDEDNESS_REFLECTION,
    CommonLineAngles,
    CommonLineTriplet,
)


def test_candidates_recover_known_geometry_and_match_matrix_solver() -> None:
    rng = np.random.default_rng(83)
    for _ in range(20):
        frames = Rotation.random(3, random_state=rng).as_matrix()
        triplet = CommonLineTriplet(
            CommonLineAngles(
                *compute_intrinsic_common_line_angles(frames[0], frames[1])
            ),
            CommonLineAngles(
                *compute_intrinsic_common_line_angles(frames[0], frames[2])
            ),
            CommonLineAngles(
                *compute_intrinsic_common_line_angles(frames[1], frames[2])
            ),
        )
        direct, mirrored = map(euler_zyz_to_matrix, alignment_euler_angles(triplet))
        truth = frames[1] @ frames[0].T
        assert (
            min(np.linalg.norm(candidate - truth) for candidate in (direct, mirrored))
            < 1e-11
        )
        expected = get_transformation_for_second_image(triplet)
        assert (
            min(
                np.max(np.abs(np.array([direct, mirrored]) - ordered))
                for ordered in (expected, expected[::-1])
            )
            < 1e-11
        )
        np.testing.assert_allclose(
            mirrored,
            HANDEDNESS_REFLECTION @ direct @ HANDEDNESS_REFLECTION,
            rtol=0,
            atol=1e-15,
        )


def test_degenerate_inconsistent_and_boundary_geometry() -> None:
    for a, b, c in [
        (0.0, 1.0, 0.5),
        (1.0, 0.0, 0.5),
        (5e-7, 1.0, 0.5),
        (1.0, 5e-7, 0.5),
        (1.0, 1.0, 2.01),
        (1.0, 1.0, np.nan),
    ]:
        triplet = CommonLineTriplet(
            CommonLineAngles(0.0, 0.0), CommonLineAngles(a, 0.0), CommonLineAngles(b, c)
        )
        with pytest.raises(ValueError):
            alignment_euler_angles(triplet)

    # At the poles, including a small tolerated cosine overshoot, both
    # handedness solutions coincide and must remain proper rotations.
    for a, b, c, expected_tilt in [
        (np.pi / 2, np.pi / 2, 0.0, 0.0),
        (np.pi / 2, np.pi / 2, np.pi, np.pi),
        (1.0, 1.0, 2.0 + 1e-8, np.pi),
    ]:
        triplet = CommonLineTriplet(
            CommonLineAngles(0.0, 0.0), CommonLineAngles(a, 0.0), CommonLineAngles(b, c)
        )
        direct, mirrored = alignment_euler_angles(triplet)
        assert direct.tilt == pytest.approx(expected_tilt, abs=1e-15)
        matrices = np.stack(
            [euler_zyz_to_matrix(direct), euler_zyz_to_matrix(mirrored)]
        )
        np.testing.assert_allclose(matrices[0], matrices[1], rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            matrices @ matrices.swapaxes(-1, -2),
            np.broadcast_to(np.eye(3), matrices.shape),
            rtol=0,
            atol=1e-15,
        )
        np.testing.assert_allclose(np.linalg.det(matrices), 1.0, rtol=0, atol=1e-15)


def test_regular_graph_matches_matrix_solver_up_to_global_handedness() -> None:
    rng = np.random.default_rng(12)
    frames = Rotation.random(3, random_state=rng).as_matrix()
    first, second = compute_intrinsic_common_line_angles(
        frames[:, None], frames[None, :]
    )
    noisy_first = first + rng.normal(0, 0.04, first.shape)
    noisy_second = second + rng.normal(0, 0.04, second.shape)

    for angle_1, angle_2 in [(first, second), (noisy_first, noisy_second)]:
        actual, fallback = graph_estimation.estimate_relative_orientation_candidates(
            angle_1, angle_2, n_third=1, random_seed=4
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
        actual_graph = graph_estimation.select_consistent_handedness(
            actual, reference_pair=(0, 1)
        )
        expected_graph = graph_estimation.select_consistent_handedness(
            expected, reference_pair=(0, 1)
        )
        assert (
            min(
                np.max(np.abs(actual_graph - candidate))
                for candidate in (
                    expected_graph,
                    graph_estimation.mirror_relative_orientations_graph(expected_graph),
                )
            )
            < 1e-11
        )


@pytest.mark.parametrize("n_third", [1, 3])
def test_graph_recovers_known_frames_with_multiple_third_views(n_third: int) -> None:
    frames = Rotation.random(6, random_state=np.random.default_rng(12)).as_matrix()
    first, second = compute_intrinsic_common_line_angles(
        frames[:, None], frames[None, :]
    )
    candidates, fallback = graph_estimation.estimate_relative_orientation_candidates(
        first, second, n_third=n_third, random_seed=4
    )
    assert not fallback.any()
    graph = graph_estimation.select_consistent_handedness(
        candidates, reference_pair=(0, 1)
    )
    truth = frames[None, :] @ frames[:, None].swapaxes(-1, -2)
    error = min(
        np.max(np.abs(graph - expected))
        for expected in (
            truth,
            graph_estimation.mirror_relative_orientations_graph(truth),
        )
    )
    assert error < 1e-11


def test_graph_handles_nearly_parallel_and_degenerate_views() -> None:
    frames = Rotation.from_euler(
        "ZYZ", [[0, 0, 0], [0.3, 0.7, -0.2], [0.3001, 1.2, 0.4], [-0.8, 0.5, 0.3]]
    ).as_matrix()
    first, second = compute_intrinsic_common_line_angles(
        frames[:, None], frames[None, :]
    )
    candidates, fallback = graph_estimation.estimate_relative_orientation_candidates(
        first, second, n_third=3, random_seed=4
    )
    assert not fallback.any()
    np.testing.assert_allclose(
        candidates @ candidates.swapaxes(-1, -2),
        np.broadcast_to(np.eye(3), candidates.shape),
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_allclose(np.linalg.det(candidates), 1.0, rtol=0, atol=2e-15)
    graph = graph_estimation.select_consistent_handedness(
        candidates, reference_pair=(0, 1)
    )
    truth = frames[None, :] @ frames[:, None].swapaxes(-1, -2)
    assert (
        min(
            np.max(np.abs(graph - candidate))
            for candidate in (
                truth,
                graph_estimation.mirror_relative_orientations_graph(truth),
            )
        )
        < 1e-8
    )

    candidates, fallback = graph_estimation.estimate_relative_orientation_candidates(
        np.zeros((3, 3)), np.zeros((3, 3)), n_third=3, random_seed=4
    )
    np.testing.assert_array_equal(fallback, ~np.eye(3, dtype=bool))
    np.testing.assert_allclose(candidates, np.broadcast_to(np.eye(3), candidates.shape))
