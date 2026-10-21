"""Geometric tests for common-line angles derived from known orientations."""

import numpy as np

from common_lines.alignment import compute_transformation_matrices_from_angles
from common_lines.forward import (
    compute_intrinsic_common_line_angles,
    find_common_lines,
)


def _rotation(axis: str, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(axis)


def test_known_views_have_expected_common_line_and_intrinsic_angles():
    """Common-line angles describe Fourier-line directions, not Radon integration directions."""
    reference = np.eye(3) # image 1 has viewing direction e_z
    tilted = _rotation("z", 30) @ _rotation("y", 90) # image 2 has viewing direction -e_x (third row, i.e. the transpose of R_y(90)^T @ R_z(30)^T e_z)

    # common line should be cross(e_z, -e_x) = -e_y
    common_line = find_common_lines(reference[2], tilted[2])
    np.testing.assert_allclose(common_line, [0, -1, 0], atol=1e-12)

    # in-image angles:
    # for image 1, -e_y has angle -90º
    # for image 2, -e_y maps to R_z(30º) @ -e_y, which has angle -90 + 30 = -60
    angle_reference, angle_tilted = compute_intrinsic_common_line_angles(
        reference, tilted
    )
    np.testing.assert_allclose(
        [angle_reference, angle_tilted],
        np.deg2rad([-90, -60]),
        atol=1e-12,
    )


def test_angles_reconstruct_the_original_or_mirrored_orientations():
    """Ground-truth common lines must invert through angular reconstitution."""
    reference = np.eye(3)
    second = _rotation("z", 24) @ _rotation("y", 39)
    third = _rotation("x", -43) @ _rotation("z", -31)

    a12, b12 = compute_intrinsic_common_line_angles(reference, second)
    a13, b13 = compute_intrinsic_common_line_angles(reference, third)
    a23, b23 = compute_intrinsic_common_line_angles(second, third)

    direct, mirrored = compute_transformation_matrices_from_angles(
        *np.rad2deg([a12, b12, a13, b13, a23, b23])
    )
    truth = np.stack([reference, second, third])

    assert any(
        np.allclose(np.stack(candidate), truth, atol=1e-10)
        for candidate in (direct, mirrored)
    )


def test_batched_parallel_and_nonparallel_views_give_valid_in_plane_directions():
    """Coincident planes have no unique common line, but must return a valid in-plane direction."""
    references = np.broadcast_to(np.eye(3), (3, 3, 3))
    others = np.stack([np.eye(3), _rotation("x", 180), _rotation("y", 90)])

    lines = find_common_lines(references[:, 2, :], others[:, 2, :])
    
    # common lines have unit norm
    np.testing.assert_allclose(np.linalg.norm(lines, axis=-1), 1, atol=1e-12)
    
    # common lines are orthogonal to viewing directions
    np.testing.assert_allclose(np.sum(lines * references[:, 2, :], axis=-1), 0, atol=1e-12)
    np.testing.assert_allclose(np.sum(lines * others[:, 2, :], axis=-1), 0, atol=1e-12)

    # Intrinsic angles agree with the common-line coordinates in each image frame
    first_angles, second_angles = compute_intrinsic_common_line_angles(references, others)
    for matrices, angles in ((references, first_angles), (others, second_angles)):
        projected = np.einsum("bij,bj->bi", matrices, lines)
        np.testing.assert_allclose(np.cos(angles), projected[:, 0], atol=1e-12)
        np.testing.assert_allclose(np.sin(angles), projected[:, 1], atol=1e-12)


def test_common_line_angles_are_independent_of_reference_frame():
    """Changing the global 3D reference frame must preserve intrinsic angles."""
    first = _rotation("z", 24) @ _rotation("y", 39)
    second = _rotation("x", -43) @ _rotation("z", -31)

    line = find_common_lines(first[2], second[2])
    angles = np.array(compute_intrinsic_common_line_angles(first, second))

    # Express both orientations in a different common reference frame
    Q = _rotation("y", 51) @ _rotation("z", 17)
    first_new = first @ Q.T
    second_new = second @ Q.T

    line_new = find_common_lines(first_new[2], second_new[2])
    angles_new = np.array(
        compute_intrinsic_common_line_angles(first_new, second_new)
    )

    # The 3D common line changes coordinates with the reference frame
    np.testing.assert_allclose(line_new, Q @ line, atol=1e-12)

    # Its angles within the two images remain unchanged
    np.testing.assert_allclose(np.cos(angles_new), np.cos(angles), atol=1e-12)
    np.testing.assert_allclose(np.sin(angles_new), np.sin(angles), atol=1e-12)
