"""Tests for orientation recovery from three pairwise common-line detections."""

from types import SimpleNamespace

import numpy as np
import pytest

from common_lines.alignment import compute_transformation_matrices


def _common_line(angle_1: float, angle_2: float):
    """Small stand-in for CommonLineResult with only the fields used here."""
    return SimpleNamespace(angle_1=angle_1, angle_2=angle_2)


def _line_from_angle(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array([np.cos(angle), np.sin(angle)])


def _same_unoriented_line(u: np.ndarray, v: np.ndarray, atol: float = 1e-10) -> bool:
    """Common lines are axes, so u and -u represent the same detected line."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return abs(np.dot(u, v)) == pytest.approx(1.0, abs=atol)


def _assert_rotation_matrix(R: np.ndarray):
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


def test_recovers_rotations_consistent_with_all_common_lines():
    # A simple, non-degenerate configuration. The three angles are not meant to
    # come from images directly; they are the pairwise common-line directions
    # after detection.
    cl12 = _common_line(angle_1=20.0, angle_2=70.0)
    cl13 = _common_line(angle_1=110.0, angle_2=35.0)
    cl23 = _common_line(angle_1=145.0, angle_2=125.0)

    R1, R2, R3 = compute_transformation_matrices(cl12, cl13, cl23)

    _assert_rotation_matrix(R1)
    _assert_rotation_matrix(R2)
    _assert_rotation_matrix(R3)
    assert np.allclose(R1, np.eye(3))

    c12 = np.array([*_line_from_angle(cl12.angle_1), 0.0])
    c13 = np.array([*_line_from_angle(cl13.angle_1), 0.0])
    c23 = np.cross(R2[2], R3[2])
    c23 = c23 / np.linalg.norm(c23)

    assert _same_unoriented_line((R1 @ c12)[:2], _line_from_angle(cl12.angle_1))
    assert _same_unoriented_line((R2 @ c12)[:2], _line_from_angle(cl12.angle_2))

    assert _same_unoriented_line((R1 @ c13)[:2], _line_from_angle(cl13.angle_1))
    assert _same_unoriented_line((R3 @ c13)[:2], _line_from_angle(cl13.angle_2))

    assert _same_unoriented_line((R2 @ c23)[:2], _line_from_angle(cl23.angle_1))
    assert _same_unoriented_line((R3 @ c23)[:2], _line_from_angle(cl23.angle_2))


def test_recovers_known_configuration_up_to_handedness():
    # This example is generated from two known passive rotation matrices. Since
    # common-line geometry cannot determine absolute handedness, the test checks
    # the pairwise angles between projection directions instead of requiring a
    # particular mirror solution.
    cl12 = _common_line(angle_1=0.0, angle_2=0.0)
    cl13 = _common_line(angle_1=90.0, angle_2=0.0)
    cl23 = _common_line(angle_1=90.0, angle_2=90.0)

    R1, R2, R3 = compute_transformation_matrices(cl12, cl13, cl23)

    normals = np.stack([R1[2], R2[2], R3[2]])
    pairwise_dot_products = normals @ normals.T

    assert np.allclose(np.diag(pairwise_dot_products), 1.0)
    assert pairwise_dot_products[0, 1] == pytest.approx(0.0, abs=1e-10)
    assert pairwise_dot_products[0, 2] == pytest.approx(0.0, abs=1e-10)
    assert pairwise_dot_products[1, 2] == pytest.approx(0.0, abs=1e-10)


def test_rejects_degenerate_common_lines_in_reference_image():
    # If the two common lines seen in image 1 are parallel, image 1 does not fix
    # a valid 3D frame and the formula would divide by sin(angle_13 - angle_12).
    cl12 = _common_line(angle_1=30.0, angle_2=10.0)
    cl13 = _common_line(angle_1=30.0, angle_2=80.0)
    cl23 = _common_line(angle_1=120.0, angle_2=45.0)

    with pytest.raises(ValueError, match="degenerate|parallel|sin"):
        compute_transformation_matrices(cl12, cl13, cl23)