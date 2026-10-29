from collections.abc import Iterable
from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
HANDEDNESS_REFLECTION = np.diag([1.0, 1.0, -1.0])
RotationTriplet = tuple[FloatArray, FloatArray, FloatArray]
HandednessPair = tuple[RotationTriplet, RotationTriplet]
RotationPair = tuple[FloatArray, FloatArray]


@dataclass(frozen=True, slots=True)
class CommonLineAngles:
    """One oriented common-line correspondence in radians.

    ``first_rad`` and ``second_rad`` describe the same oriented 3D direction
    in the first and second image planes, respectively. Any detector-reversal
    correction must already be included. First and second refer to the ordered
    endpoints of this pair, not to fixed image numbers in a triplet.
    Scalars and broadcast-compatible arrays are accepted by the solvers.
    """

    first_rad: float | np.ndarray
    second_rad: float | np.ndarray


@dataclass(frozen=True, slots=True)
class CommonLineTriplet:
    """Oriented common lines connecting three images, in radians.

    The pairs are ordered 1 -> 2, 1 -> 3, and 2 -> 3. Image 1 defines the
    reference coordinate system. All six angle fields broadcast together.
    """

    one_with_two: CommonLineAngles
    one_with_three: CommonLineAngles
    two_with_three: CommonLineAngles

    def angular_differences(
        self,
    ) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        """Return signed separations in images 1, 2, and 3, without wrapping.

        Subtraction orders are line 1-3 minus 1-2 in image 1, line 2-3 minus
        1-2 in image 2, and line 2-3 minus 1-3 in image 3.
        """
        return (
            self.one_with_three.first_rad - self.one_with_two.first_rad,
            self.two_with_three.first_rad - self.one_with_two.second_rad,
            self.two_with_three.second_rad - self.one_with_three.second_rad,
        )


def _prepare_triplet(triplet: CommonLineTriplet) -> CommonLineTriplet:
    """
    Broadcast the six angle fields as float64 arrays without changing inputs.

    Real, finite angles are assumed; NumPy reports incompatible shapes.
    """
    angles = (
        triplet.one_with_two.first_rad,
        triplet.one_with_two.second_rad,
        triplet.one_with_three.first_rad,
        triplet.one_with_three.second_rad,
        triplet.two_with_three.first_rad,
        triplet.two_with_three.second_rad,
    )
    a12, b12, a13, b13, a23, b23 = np.broadcast_arrays(
        *(np.asarray(a, dtype=np.float64) for a in angles)
    )
    return CommonLineTriplet(
        CommonLineAngles(a12, b12),
        CommonLineAngles(a13, b13),
        CommonLineAngles(a23, b23),
    )


def compute_transformation_matrices(
    triplet: CommonLineTriplet, eps: float = 1.0e-6
) -> HandednessPair:
    """
    Recover both handedness solutions for three reference-to-image frames.

    Return ``(direct, mirrored)``, each branch a tuple ``(R1, R2, R3)`` satisfying
    ``v_i = R_i @ v_ref``, with image 1 fixed to the identity. Matrix shapes
    are ``broadcast_shape + (3, 3)``; scalar inputs return 3x3 matrices.
    Angles are oriented radians and need not be wrapped.

    ``eps`` retains the sine and squared-z tolerances of angular reconstitution.
    Any geometrically invalid entry raises ValueError for the whole call.
    Real, finite input angles are assumed. Empty batches are accepted.
    """
    triplet = _prepare_triplet(triplet)
    c12, c13, c23 = _get_common_lines_from_angles(triplet, eps)
    r2 = _compute_matrix_from_common_lines(
        triplet.one_with_two.second_rad,
        c12,
        triplet.two_with_three.first_rad,
        c23,
        eps=eps,
    )
    r3 = _compute_matrix_from_common_lines(
        triplet.one_with_three.second_rad,
        c13,
        triplet.two_with_three.second_rad,
        c23,
        eps=eps,
    )
    r1 = np.broadcast_to(np.eye(3), r2.shape).copy()
    direct = (r1, r2, r3)
    branches = direct, mirror_transformation_matrices(direct)
    return branches


def _get_common_lines_from_angles(
    triplet: CommonLineTriplet, eps: float = 1.0e-6
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Recover the common-line directions in the reference coordinate system
    from prepared, broadcasted angles.

    Image 1 defines the reference coordinate system, so its image plane is the
    reference XY plane. The input angles are expressed in radians.
    """
    a, b, c = triplet.angular_differences()
    shape = np.shape(a)

    c12 = np.zeros(shape + (3,), dtype=np.float64)
    c13 = np.zeros_like(c12)
    c23 = np.zeros_like(c12)

    c12[..., 0] = np.cos(triplet.one_with_two.first_rad)
    c12[..., 1] = np.sin(triplet.one_with_two.first_rad)
    c13[..., 0] = np.cos(triplet.one_with_three.first_rad)
    c13[..., 1] = np.sin(triplet.one_with_three.first_rad)

    sin_a = np.sin(a)
    if np.any(np.abs(sin_a) < eps):
        raise ValueError(
            "Degenerate configuration: common lines in image 1 are parallel."
        )
    # Dot products give c23.c12 = cos(b), c23.c13 = cos(c).

    c23[..., 0] = (c13[..., 1] * np.cos(b) - c12[..., 1] * np.cos(c)) / sin_a
    c23[..., 1] = (c12[..., 0] * np.cos(c) - c13[..., 0] * np.cos(b)) / sin_a

    z2 = 1.0 - c23[..., 0] ** 2 - c23[..., 1] ** 2
    if np.any(z2 < -eps):
        raise ValueError("Inconsistent common-line angles: no valid 3D configuration.")

    c23[..., 2] = np.sqrt(np.maximum(z2, 0.0))

    return c12, c13, c23


def get_transformation_for_second_image(
    triplet: CommonLineTriplet, eps: float = 1.0e-6
) -> RotationPair:
    """Return direct and mirrored image-1-to-image-2 matrices per triplet.

    Each result satisfies ``v_2 = R2 @ v_1`` and has shape
    ``broadcast_shape + (3, 3)``. No aggregation or rotation projection is applied.
    Only image 2 is reconstructed, so degeneracy confined to reconstructing
    image 3 does not invalidate this estimate.

    Return ``(direct, mirrored)``. Any geometrically invalid entry raises
    ValueError for the whole call. Real, finite input angles are assumed.
    Empty batches are accepted.
    """
    triplet = _prepare_triplet(triplet)
    c12, _, c23 = _get_common_lines_from_angles(triplet, eps)
    r2 = _compute_matrix_from_common_lines(
        triplet.one_with_two.second_rad,
        c12,
        triplet.two_with_three.first_rad,
        c23,
        eps=eps,
    )
    branches = r2, HANDEDNESS_REFLECTION @ r2 @ HANDEDNESS_REFLECTION
    return branches


@overload
def mirror_transformation_matrices(rotations: RotationTriplet) -> RotationTriplet: ...


@overload
def mirror_transformation_matrices(
    rotations: Iterable[np.ndarray],
) -> tuple[FloatArray, ...]: ...


def mirror_transformation_matrices(
    rotations: Iterable[np.ndarray],
) -> tuple[FloatArray, ...]:
    """Return opposite-handed frames via J @ R @ J, J = diag(1, 1, -1).

    Matrices may have arbitrary leading batch dimensions. Each matrix maps
    coordinates from the common reference frame to its image frame.
    """
    J = HANDEDNESS_REFLECTION
    return tuple(J @ rotation @ J for rotation in rotations)


def _compute_matrix_from_common_lines(
    theta_a_rad: float | np.ndarray,
    c_a: FloatArray,
    theta_b_rad: float | np.ndarray,
    c_b: FloatArray,
    *,
    eps: float,
) -> FloatArray:
    """Solve reference-to-image rows from two lines, preserving trailing axes."""
    sin_diff = np.sin(theta_b_rad - theta_a_rad)
    if np.any(np.abs(sin_diff) < eps):
        raise ValueError(
            "Degenerate configuration: common lines in one image are parallel."
        )
    numerator = np.stack(
        (
            np.sin(theta_b_rad)[..., None] * c_a - np.sin(theta_a_rad)[..., None] * c_b,
            np.cos(theta_a_rad)[..., None] * c_b - np.cos(theta_b_rad)[..., None] * c_a,
            np.cross(c_a, c_b),
        ),
        axis=-2,
    )
    # The last row is the projection direction, orthogonal to both lines.
    return numerator / sin_diff[..., None, None]
