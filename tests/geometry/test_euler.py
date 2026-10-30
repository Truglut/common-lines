"""Tests for the passive ZYZ convention and batched Euler conversion."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common_lines.geometry.euler import EulerAngles, euler_zyz_to_matrix
from common_lines.reconstitution.triplets import HANDEDNESS_REFLECTION


def test_matches_independent_rotation_composition() -> None:
    rng = np.random.default_rng(78)
    angles = rng.uniform(-2 * np.pi, 2 * np.pi, size=(32, 3))
    # Include identity and singular/nearly singular tilts alongside generic views.
    angles[:5] = [
        [0, 0, 0],
        [0.3, 0, -0.7],
        [0.3, np.pi, -0.7],
        [0.3, 1e-12, -0.7],
        [0.3, np.pi - 1e-12, -0.7],
    ]

    result = euler_zyz_to_matrix(EulerAngles(*angles.T))
    # SciPy's intrinsic ZYZ rotation is active; its transpose changes coordinates.
    expected = Rotation.from_euler("ZYZ", angles).as_matrix().swapaxes(-1, -2)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        result @ result.swapaxes(-1, -2),
        np.broadcast_to(np.eye(3), result.shape),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(np.linalg.det(result), 1.0, rtol=0, atol=1e-15)


def test_viewing_direction_is_independent_of_in_plane_rotation() -> None:
    rot, tilt = 0.73, 1.19
    matrices = euler_zyz_to_matrix(EulerAngles(rot, tilt, np.array([-1.2, 0.0, 0.8])))
    direction = np.array(
        [np.sin(tilt) * np.cos(rot), np.sin(tilt) * np.sin(rot), np.cos(tilt)]
    )
    np.testing.assert_allclose(
        matrices[..., 2, :], np.broadcast_to(direction, (3, 3)), rtol=0, atol=1e-15
    )
    # The reference-space viewing direction becomes the image's positive z axis.
    np.testing.assert_allclose(
        matrices @ direction, np.broadcast_to([0, 0, 1], (3, 3)), rtol=0, atol=1e-15
    )


def test_equivalent_angles_at_poles() -> None:
    angles = np.array([[0.43, 0.0, -0.71], [0.43, np.pi, -0.71]])
    # At tilt=0 only rot+psi matters; at tilt=pi only rot-psi matters.
    equivalent = angles + [[0.6, 0.0, -0.6], [0.6, 0.0, 0.6]]
    np.testing.assert_allclose(
        euler_zyz_to_matrix(EulerAngles(*equivalent.T)),
        euler_zyz_to_matrix(EulerAngles(*angles.T)),
        rtol=0,
        atol=1e-15,
    )


def test_broadcasting_preserves_dtype_and_reuses_output_buffer() -> None:
    rot = np.array([[0.3], [-0.8]], dtype=np.float32)
    tilt = np.array([0.0, 0.7, np.pi], dtype=np.float32)
    psi = np.array(-0.2, dtype=np.float32)
    out = np.full((2, 3, 3, 3), np.nan, dtype=np.float32)

    assert euler_zyz_to_matrix(EulerAngles(rot, tilt, psi), out=out) is out
    allocated = euler_zyz_to_matrix(EulerAngles(rot, tilt, psi))
    assert allocated.shape == out.shape
    assert allocated.dtype == np.float32
    np.testing.assert_array_equal(out, allocated)
    # Broadcasting must represent the Cartesian product of rot and tilt values.
    for i, j in np.ndindex(2, 3):
        expected = Rotation.from_euler("ZYZ", [rot[i, 0], tilt[j], psi]).as_matrix().T
        np.testing.assert_allclose(out[i, j], expected, rtol=0, atol=2e-7)


def test_rejects_invalid_array_contracts() -> None:
    angles = np.array([0.3, 0.7, -0.2])
    with pytest.raises(TypeError, match="real floating-point"):
        euler_zyz_to_matrix(EulerAngles(*angles.astype(np.int64)))
    with pytest.raises(RuntimeError, match="Invalid output array"):
        euler_zyz_to_matrix(EulerAngles(*angles), out=np.empty((2, 3, 3)))
    with pytest.raises(RuntimeError, match="Invalid output array"):
        euler_zyz_to_matrix(
            EulerAngles(*angles), out=np.empty((3, 3), dtype=np.float32)
        )


def test_opposite_tilt_or_adding_pi_equals_handedness_reflection() -> None:
    rot = np.deg2rad(np.array([40.0, 80.0, 234.0]))
    tilt = np.deg2rad(np.array([10.0, 35.0, 72.0]))
    psi = np.deg2rad(np.array([0.0, 27.0, -43.0]))

    euler = EulerAngles(rot, tilt, psi)

    direct_matrix = euler_zyz_to_matrix(euler)
    manually_mirrored = HANDEDNESS_REFLECTION @ direct_matrix @ HANDEDNESS_REFLECTION

    opposite_tilt_euler = EulerAngles(rot, -tilt, psi)
    opposite_tilt_matrix = euler_zyz_to_matrix(opposite_tilt_euler)
    np.testing.assert_allclose(opposite_tilt_matrix, manually_mirrored, atol=1.0e-8)

    add_pi_euler = EulerAngles(rot + np.pi, tilt, psi + np.pi)
    add_pi_matrix = euler_zyz_to_matrix(add_pi_euler)
    np.testing.assert_allclose(add_pi_matrix, manually_mirrored, atol=1.0e-8)
