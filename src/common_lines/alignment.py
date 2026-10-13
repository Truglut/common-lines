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
    Recover the coordinate frames of three images from their common lines.

    Image 1 defines the reference coordinate system. For each image ``i``, the
    returned matrix ``R_i`` transforms coordinates from this reference system
    to the coordinate system of image ``i``:

        v_i = R_i @ v_ref

    Therefore, rows 0 and 1 of ``R_i`` are the image-plane basis vectors
    expressed in reference coordinates, and row 2 is the projection direction.
    For image 1, ``R_1`` is the identity matrix.

    Two solutions are returned because common-line geometry determines the
    orientations only up to a global handedness reflection.

    Parameters
    ----------
    common_line_result_12 : CommonLineResult
        Detected oriented common line between images 1 and 2.
    common_line_result_13 : CommonLineResult
        Detected oriented common line between images 1 and 3.
    common_line_result_23 : CommonLineResult
        Detected oriented common line between images 2 and 3.
    theta_deg : np.ndarray
        Angular grid used to generate the sinograms, in degrees.
    eps : float, optional
        Numerical tolerance used to reject degenerate or inconsistent triplets.

    Returns
    -------
    tuple
        Pair ``(direct, mirrored)``. Each element is ``(R_1, R_2, R_3)``,
        with the matrices defined by ``v_i = R_i @ v_ref``. The two solutions
        are related by conjugation with ``HANDEDNESS_REFLECTION``.

    Notes
    -----
    Common-line information alone cannot determine which of the two handedness
    solutions corresponds to the physical structure.
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
    Recover the coordinate frames of three images from oriented common-line angles.

    Image 1 defines the reference coordinate system. For each image ``i``, the
    returned matrix ``R_i`` transforms vector coordinates from that reference
    system to the coordinate system of image ``i``:

        v_i = R_i @ v_ref

    Thus, if a 3D direction has coordinates ``c_ref`` in the reference system
    and appears at angle ``theta`` in image ``i``, then

        R_i @ c_ref = [cos(theta), sin(theta), 0].

    Image 1 is fixed to ``R_1 = I``. The orientations of images 2 and 3 are
    reconstructed relative to that coordinate system using angular
    reconstitution.

    All input angles are oriented common-line angles in degrees. They may lie
    in ``[0, 360)`` even if the underlying sinograms were sampled only over
    ``[0, 180)``.

    Parameters
    ----------
    angle_12_in_1_deg, angle_12_in_2_deg
        Oriented angles, in degrees, of the common line shared by images 1 and 2,
        expressed in the coordinate systems of images 1 and 2 respectively.
    angle_13_in_1_deg, angle_13_in_3_deg
        Oriented angles, in degrees, of the common line shared by images 1 and 3,
        expressed in the coordinate systems of images 1 and 3 respectively.
    angle_23_in_2_deg, angle_23_in_3_deg
        Oriented angles, in degrees, of the common line shared by images 2 and 3,
        expressed in the coordinate systems of images 2 and 3 respectively.
    eps
        Numerical tolerance used to reject degenerate or inconsistent triplets.

    Returns
    -------
    tuple
        Pair ``(direct, mirrored)`` containing the two handedness solutions.
        Each solution is a tuple ``(R_1, R_2, R_3)`` satisfying

            v_i = R_i @ v_ref.

    Notes
    -----
    Common-line information alone cannot determine which handedness solution
    corresponds to the physical structure.
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
    Recover the common-line directions in the reference coordinate system.

    Image 1 defines the reference coordinate system, so its image plane is the
    reference XY plane. The input angles are expressed in radians.

    Returns
    -------
    tuple of numpy.ndarray
        Unit 3D directions ``(c_12, c_13, c_23)`` expressed in the coordinate
        system of image 1.
    """
    c_12 = np.zeros(3)
    c_13 = np.zeros(3)
    c_23 = np.zeros(3)

    ## Compute the 3D coordinates of the common lines
    # Image 1 defines the reference coordinate system. Therefore the common
    # lines observed in image 1 lie directly in the reference XY plane.
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


def get_transformation_for_second_image(
    angle_12_in_1_deg: float,
    angle_12_in_2_deg: float,
    angle_13_in_1_deg: float,
    angle_13_in_3_deg: float,
    angle_23_in_2_deg: float,
    angle_23_in_3_deg: float,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover the coordinate transformation for image 2 from three common-line pairs.

    Image 1 defines the reference coordinate system. The returned matrix ``R_2``
    transforms coordinates from the system of image 1 to the system of image 2:

        v_2 = R_2 @ v_1

    because the coordinate system of image 1 is fixed to the identity.

    Only the matrix for image 2 is computed. This allows the function to be used
    when image 2 can be reconstructed from a triplet even if the full orientation
    of image 3 is not recoverable.

    All input common-line angles are oriented (i.e., in the [0, 360) interval) angles
    in degrees.

    Returns
    -------
    tuple of numpy.ndarray
        ``(direct_R2, mirrored_R2)``, the two handedness candidates for the
        transformation from image 1 coordinates to image 2 coordinates.
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
    Return the opposite-handed coordinate-frame solution.

    Each input matrix ``R_i`` is assumed to transform coordinates from a common
    reference system to image ``i``. The opposite-handed solution is obtained as

        J @ R_i @ J

    with ``J = diag(1, 1, -1)``.

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
    Construct the coordinate transformation for one image from two common lines.

    The vectors ``c_a`` and ``c_b`` are the coordinates of two 3D common-line
    directions in the reference coordinate system. ``theta_a_rad`` and
    ``theta_b_rad`` are the angles of those same directions in the image
    coordinate system.

    The returned matrix ``R`` satisfies

        R @ c_a = [cos(theta_a_rad), sin(theta_a_rad), 0]
        R @ c_b = [cos(theta_b_rad), sin(theta_b_rad), 0]

    and therefore transforms coordinates from the reference system to the image
    coordinate system.

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
        Proper 3D rotation matrix transforming reference coordinates into the
        coordinate system of the image.
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
    rotation_matrix[0, :] = (
        np.sin(theta_b_rad) * c_a - np.sin(theta_a_rad) * c_b
    ) / sin_diff
    rotation_matrix[1, :] = (
        np.cos(theta_a_rad) * c_b - np.cos(theta_b_rad) * c_a
    ) / sin_diff

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
