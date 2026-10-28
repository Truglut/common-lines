"""Tests for the passive ZYZ convention and batched Euler conversion."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common_lines.euler import euler_zyz_to_matrix


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

    result = euler_zyz_to_matrix(*angles.T)
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
    matrices = euler_zyz_to_matrix(
        np.array(rot), np.array(tilt), np.array([-1.2, 0.0, 0.8])
    )
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
        euler_zyz_to_matrix(*equivalent.T),
        euler_zyz_to_matrix(*angles.T),
        rtol=0,
        atol=1e-15,
    )


def test_broadcasting_preserves_dtype_and_reuses_output_buffer() -> None:
    rot = np.array([[0.3], [-0.8]], dtype=np.float32)
    tilt = np.array([0.0, 0.7, np.pi], dtype=np.float32)
    psi = np.array(-0.2, dtype=np.float32)
    out = np.full((2, 3, 3, 3), np.nan, dtype=np.float32)

    assert euler_zyz_to_matrix(rot, tilt, psi, out=out) is out
    allocated = euler_zyz_to_matrix(rot, tilt, psi)
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
        euler_zyz_to_matrix(*angles.astype(np.int64))
    with pytest.raises(RuntimeError, match="Invalid output array"):
        euler_zyz_to_matrix(*angles, out=np.empty((2, 3, 3)))
    with pytest.raises(RuntimeError, match="Invalid output array"):
        euler_zyz_to_matrix(*angles, out=np.empty((3, 3), dtype=np.float32))
