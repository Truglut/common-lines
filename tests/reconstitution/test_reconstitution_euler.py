"""Geometric correctness and regular-case comparisons with matrix alignment."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common_lines.geometry.euler import euler_zyz_to_matrix
from common_lines.geometry.forward import compute_intrinsic_common_line_angles
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
