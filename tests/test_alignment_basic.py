"""Tests for orientation recovery from three pairwise common-line detections."""

import numpy as np
import pytest

from common_lines.alignment import (
    HANDEDNESS_REFLECTION,
    compute_transformation_matrices_from_angles,
)



def _image_plane_direction(angle_deg: float) -> np.ndarray:
    """Return the oriented image-plane direction associated with an angle."""
    angle = np.deg2rad(angle_deg)
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def _assert_rotation_matrix(R: np.ndarray):
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


def _assert_common_line_correspondence(
    R_a: np.ndarray,
    R_b: np.ndarray,
    angle_a: float,
    angle_b: float,
) -> None:
    """Check one oriented common-line correspondence between two image frames."""
    direction_a = _image_plane_direction(angle_a)
    direction_b = _image_plane_direction(angle_b)

    # Transformation matrices map global coordinates into image coordinates.
    # Recover the global common line from image a and express it in image b.
    common_line_global = R_a.T @ direction_a

    assert np.allclose(R_a @ common_line_global, direction_a, atol=1e-10)
    assert np.allclose(R_b @ common_line_global, direction_b, atol=1e-10)


def test_recovers_both_handednesses_consistent_with_all_common_lines():
    # A simple, non-degenerate configuration. The angles represent oriented
    # common-line directions, including any 180-degree correction introduced
    # when detection finds a reversed profile.
    angles = {
        "angle_12_in_1_deg": 20.0,
        "angle_12_in_2_deg": 70.0,
        "angle_13_in_1_deg": 110.0,
        "angle_13_in_3_deg": 35.0,
        "angle_23_in_2_deg": 145.0,
        "angle_23_in_3_deg": 125.0,
    }

    direct, mirrored = compute_transformation_matrices_from_angles(**angles)

    for R1, R2, R3 in (direct, mirrored):
        _assert_rotation_matrix(R1)
        _assert_rotation_matrix(R2)
        _assert_rotation_matrix(R3)
        assert np.allclose(R1, np.eye(3))

        _assert_common_line_correspondence(
            R1, R2, 20.0, 70.0
        )
        _assert_common_line_correspondence(
            R1, R3, 110.0, 35.0
        )
        _assert_common_line_correspondence(
            R2, R3, 145.0, 125.0
        )

    J = HANDEDNESS_REFLECTION
    for R_direct, R_mirrored in zip(direct, mirrored):
        assert np.allclose(R_mirrored, J @ R_direct @ J, atol=1e-10)


def test_recovers_known_configuration_in_both_handednesses():
    # These angles describe three mutually orthogonal projection directions.
    # Both returned branches must preserve that geometry and be related by the
    # global handedness reflection.
    direct, mirrored = compute_transformation_matrices_from_angles(
        angle_12_in_1_deg=0.0,
        angle_12_in_2_deg=0.0,
        angle_13_in_1_deg=90.0,
        angle_13_in_3_deg=0.0,
        angle_23_in_2_deg=90.0,
        angle_23_in_3_deg=90.0,
    )

    for rotations in (direct, mirrored):
        for R in rotations:
            _assert_rotation_matrix(R)

        normals = np.stack([R[2] for R in rotations])
        pairwise_dot_products = normals @ normals.T

        assert np.allclose(pairwise_dot_products, np.eye(3), atol=1e-10)

    J = HANDEDNESS_REFLECTION
    for R_direct, R_mirrored in zip(direct, mirrored):
        assert np.allclose(R_mirrored, J @ R_direct @ J, atol=1e-10)


def test_rejects_degenerate_common_lines_in_reference_image():
    # If the two common lines seen in image 1 are parallel, image 1 does not fix
    # a valid 3D frame and the formula would divide by sin(angle_13 - angle_12).
    with pytest.raises(ValueError, match="degenerate|parallel|sin"):
        compute_transformation_matrices_from_angles(
            angle_12_in_1_deg=30.0,
            angle_12_in_2_deg=10.0,
            angle_13_in_1_deg=30.0,
            angle_13_in_3_deg=80.0,
            angle_23_in_2_deg=120.0,
            angle_23_in_3_deg=45.0,
        )