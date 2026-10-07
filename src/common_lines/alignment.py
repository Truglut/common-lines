from typing import Iterable, overload

import numpy as np
from common_lines.detection import CommonLineResult

HANDEDNESS_REFLECTION = np.diag([1.0, 1.0, -1.0])
RotationTriplet = tuple[np.ndarray, np.ndarray, np.ndarray]
HandednessPair = tuple[RotationTriplet, RotationTriplet]


def compute_transformation_matrices(
    common_line_result_12: CommonLineResult,
    common_line_result_13: CommonLineResult,
    common_line_result_23: CommonLineResult,
    theta_deg: np.ndarray,
    eps: float = 1.0e-6,
) -> HandednessPair:
    """
    Compute both handedness solutions for the orientations of three images.

    The returned matrices follow the passive / inverse rotation convention. Rows
    0 and 1 contain the image axes in the reference coordinate system, while row
    2 contains the projection direction. Image 1 is fixed to the identity.

    Parameters
    ----------
    common_line_result_12 : CommonLineResult
        Detected oriented common line between images 1 and 2.
    common_line_result_13 : CommonLineResult
        Detected oriented common line between images 1 and 3.
    common_line_result_23 : CommonLineResult
        Detected oriented common line between images 2 and 3.
    theta_deg : np.ndarray
        Angular grid used to generate the sinograms from which the common lines
        were detected, in degrees.
    eps : float, optional
        Numerical tolerance used to reject degenerate or inconsistent triplets.

    Returns
    -------
    tuple
        Pair ``(direct, mirrored)``. Each element is a tuple containing the
        transformation matrices for images 1, 2, and 3. The two solutions are
        related by conjugation with ``HANDEDNESS_REFLECTION``.

    Notes
    -----
    Common-line information cannot distinguish between these two globally
    mirrored configurations. Neither returned branch should be treated as the
    physically correct hand without additional information.
    """
    # Extract the angles of the common lines from the results
    angle_12_in_1, angle_12_in_2 = common_line_result_12.oriented_angles(
        theta_1=theta_deg, theta_2=theta_deg
    )
    angle_13_in_1, angle_13_in_3 = common_line_result_13.oriented_angles(
        theta_1=theta_deg, theta_2=theta_deg
    )
    angle_23_in_2, angle_23_in_3 = common_line_result_23.oriented_angles(
        theta_1=theta_deg, theta_2=theta_deg
    )

    return compute_transformation_matrices_from_angles(
        angle_12_in_1_deg=float(angle_12_in_1),
        angle_12_in_2_deg=float(angle_12_in_2),
        angle_13_in_1_deg=float(angle_13_in_1),
        angle_13_in_3_deg=float(angle_13_in_3),
        angle_23_in_2_deg=float(angle_23_in_2),
        angle_23_in_3_deg=float(angle_23_in_3),
        eps=eps,
    )


def compute_transformation_matrices_from_angles(
    angle_12_in_1_deg: float,
    angle_12_in_2_deg: float,
    angle_13_in_1_deg: float,
    angle_13_in_3_deg: float,
    angle_23_in_2_deg: float,
    angle_23_in_3_deg: float,
    eps: float = 1.0e-6,
) -> HandednessPair:
    """
    Compute both handedness solutions from oriented common-line angles.

    Angles are expressed in degrees and describe oriented image-plane directions.
    They may therefore lie in the range [0, 360), even when the sinograms were
    computed only for angles in [0, 180).

    The returned transformation matrices follow the passive / inverse rotation convention,
    where the rotation is applied to the coordinate system rather than the image itself.
    Equivalently, the transformation matrices can be interpreted as the rotation that
    has to be applied to the images to align them with the reference.

    The computation is done following a method almost identical to the one described in
    "Angular reconstitution: A posteriori assignment of projection directions
    for 3D reconstruction" (1987), by Van Heel.
    The main difference is that we fix the rotation of image 1 to the identity matrix,
    and compute the rotations of images 2 and 3 relative to image 1.

    Parameters
    ----------
    angle_12_in_1, angle_12_in_2
        Oriented common-line angles for images 1 and 2, in degrees.
    angle_13_in_1, angle_13_in_3
        Oriented common-line angles for images 1 and 3, in degrees.
    angle_23_in_2, angle_23_in_3
        Oriented common-line angles for images 2 and 3, in degrees.
    eps
        Numerical tolerance used to reject degenerate or inconsistent triplets.

    Returns
    -------
    tuple
        Pair ``(direct, mirrored)`` containing the two globally consistent
        handedness solutions for the triplet. Each solution contains the
        transformation matrices for images 1, 2, and 3.

    Notes
    -----
    Common-line information cannot distinguish between these two globally
    mirrored configurations. Neither returned branch should be treated as the
    physically correct hand without additional information.
    """
    # Every angle from this point on should be in radians
    angle_12_in_1 = _angle_to_rad(angle_12_in_1_deg, "angle_12_in_1")
    angle_12_in_2 = _angle_to_rad(angle_12_in_2_deg, "angle_12_in_2")
    angle_13_in_1 = _angle_to_rad(angle_13_in_1_deg, "angle_13_in_1")
    angle_13_in_3 = _angle_to_rad(angle_13_in_3_deg, "angle_13_in_3")
    angle_23_in_2 = _angle_to_rad(angle_23_in_2_deg, "angle_23_in_2")
    angle_23_in_3 = _angle_to_rad(angle_23_in_3_deg, "angle_23_in_3")

    c_12, c_13, c_23 = _get_common_lines_from_angles(
        angle_12_in_1,
        angle_12_in_2,
        angle_13_in_1,
        angle_13_in_3,
        angle_23_in_2,
        angle_23_in_3,
        eps,
    )

    # Compute the rotation matrices for each image based on the detected angles
    R1 = np.eye(3)  # Image 1 is fixed to the identity matrix
    R2 = _compute_matrix_from_common_lines(
        angle_12_in_2, c_12, angle_23_in_2, c_23, eps=eps
    )
    R3 = _compute_matrix_from_common_lines(
        angle_13_in_3, c_13, angle_23_in_3, c_23, eps=eps
    )

    direct = (R1, R2, R3)
    mirrored = mirror_transformation_matrices(direct)

    return direct, mirrored


def _get_common_lines_from_angles(
    angle_12_in_1_rad: float,
    angle_12_in_2_rad: float,
    angle_13_in_1_rad: float,
    angle_13_in_3_rad: float,
    angle_23_in_2_rad: float,
    angle_23_in_3_rad: float,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets the 3D coordinates of the common lines among three images from their measured
    angles, in radians.
    """
    c_12 = np.zeros(3)
    c_13 = np.zeros(3)
    c_23 = np.zeros(3)

    ## Compute the 3D coordinates of the common lines
    # The common lines contained in image 1 lie in the XY plane because
    # image 1 is fixed to the identity matrix
    c_12[0] = np.cos(angle_12_in_1_rad)
    c_12[1] = np.sin(angle_12_in_1_rad)
    c_13[0] = np.cos(angle_13_in_1_rad)
    c_13[1] = np.sin(angle_13_in_1_rad)

    # The common line between image 2 and 3 is computed using the relative angles
    # between the three common lines
    a = angle_13_in_1_rad - angle_12_in_1_rad
    b = angle_23_in_2_rad - angle_12_in_2_rad
    c = angle_23_in_3_rad - angle_13_in_3_rad

    # Check for degenerate geometric configurations
    sin_a = np.sin(a)
    if abs(sin_a) < eps:
        raise ValueError(
            "Degenerate configuration: common lines in image 1 are parallel."
        )

    # cos b = c_23 · c_12
    # cos c = c_23 · c_13
    c_23[0] = (c_13[1] * np.cos(b) - c_12[1] * np.cos(c)) / sin_a
    c_23[1] = (c_12[0] * np.cos(c) - c_13[0] * np.cos(b)) / sin_a

    # Use c_23 is unitary to compute the last coordinate
    z2 = 1.0 - c_23[0] ** 2 - c_23[1] ** 2
    if z2 < -eps:
        raise ValueError("Inconsistent common-line angles: no valid 3D configuration.")
    c_23[2] = np.sqrt(max(z2, 0.0))

    return c_12, c_13, c_23


def get_alignment_for_second_image(
    angle_12_in_1_deg: float,
    angle_12_in_2_deg: float,
    angle_13_in_1_deg: float,
    angle_13_in_3_deg: float,
    angle_23_in_2_deg: float,
    angle_23_in_3_deg: float,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute both handedness solutions from oriented common-line angles only for the
    second image ("image 2"). This is useful for cases where the geometric
    configuration allows for calculating the alignment of image 2, but not image 3.
    Input angles are in degrees.
    """
    # Every angle from this point on should be in radians
    angle_12_in_1 = _angle_to_rad(angle_12_in_1_deg, "angle_12_in_1")
    angle_12_in_2 = _angle_to_rad(angle_12_in_2_deg, "angle_12_in_2")
    angle_13_in_1 = _angle_to_rad(angle_13_in_1_deg, "angle_13_in_1")
    angle_13_in_3 = _angle_to_rad(angle_13_in_3_deg, "angle_13_in_3")
    angle_23_in_2 = _angle_to_rad(angle_23_in_2_deg, "angle_23_in_2")
    angle_23_in_3 = _angle_to_rad(angle_23_in_3_deg, "angle_23_in_3")

    c_12, _, c_23 = _get_common_lines_from_angles(
        angle_12_in_1,
        angle_12_in_2,
        angle_13_in_1,
        angle_13_in_3,
        angle_23_in_2,
        angle_23_in_3,
        eps,
    )

    # Compute the rotation matrix only for image 2
    direct_r2 = _compute_matrix_from_common_lines(
        angle_12_in_2, c_12, angle_23_in_2, c_23, eps=eps
    )
    mirrored_r2 = HANDEDNESS_REFLECTION @ direct_r2 @ HANDEDNESS_REFLECTION

    return direct_r2, mirrored_r2


@overload
def mirror_transformation_matrices(rotations: RotationTriplet) -> RotationTriplet: ...


@overload
def mirror_transformation_matrices(
    rotations: Iterable[np.ndarray],
) -> tuple[np.ndarray, ...]: ...


def mirror_transformation_matrices(
    rotations: Iterable[np.ndarray],
) -> tuple[np.ndarray, ...]:
    """
    Return the opposite-handed version of a triplet of transformation matrices.

    Parameters
    ----------
    rotations: Iterable[np.ndarray]
        Transformation matrices for three images in a common reference frame.

    Returns
    -------
    tuple of numpy.ndarray
        Matrices obtained as ``J @ R @ J`` for
        ``J = HANDEDNESS_REFLECTION``.
    """
    J = HANDEDNESS_REFLECTION
    return tuple(J @ rotation @ J for rotation in rotations)


def _compute_matrix_from_common_lines(
    theta_a_rad: float,
    c_a: np.ndarray,
    theta_b_rad: float,
    c_b: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    """
    Computes the transformation matrix corresponding to an image from the common lines
    it shares with two other images, using the precomputed 3D coordinates of these
    given common lines.

    Parameters
    ----------
    theta_a_rad : float
        Detected angle of the first common line in the image, in radians.
    c_a : np.ndarray
        3D coordinates of the first common line.
    theta_b_rad : float
        Detected angle of the second common line in the image, in radians.
    c_b : np.ndarray
        3D coordinates of the second common line.
    eps : float
        Numerical tolerance used to reject parallel image-plane directions.

    Returns
    -------
    np.ndarray
        3x3 corresponding to the rotation matrix that aligns the image to the reference
        (the reference is implicit in the 3D coordinates of the common lines).
    """
    # Read angle of both common lines with image axes and compute difference
    sin_diff = np.sin(theta_b_rad - theta_a_rad)
    if abs(sin_diff) < eps:
        raise ValueError(
            "Degenerate configuration: common lines in one image are parallel."
        )

    rotation_matrix = np.zeros((3, 3), dtype=np.float64)

    # The first two rows of the rotation matrix are obtained by solving a linear system
    # given by the coordinates of the two common lines
    rotation_matrix[0, :] = (np.sin(theta_b_rad) * c_a - np.sin(theta_a_rad) * c_b) / sin_diff
    rotation_matrix[1, :] = (np.cos(theta_a_rad) * c_b - np.cos(theta_b_rad) * c_a) / sin_diff

    # The last row is the projection direction: orthogonal to both common lines, and
    # unitary
    rotation_matrix[2, :] = np.cross(c_a, c_b) / sin_diff

    return rotation_matrix


def _angle_to_rad(angle: float | np.ndarray | None, name: str) -> float:
    """Convert a scalar angle in degrees to radians."""
    if angle is None:
        raise ValueError(
            f"{name} cannot be None. Pass theta arrays to detect_common_lines."
        )

    angle = np.asarray(angle)

    if angle.size != 1:
        raise ValueError(
            f"{name} must contain a single angle. Got shape {angle.shape}."
        )

    return np.deg2rad(float(angle.reshape(-1)[0]))
