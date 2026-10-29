from dataclasses import dataclass

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


def broadcast_triplet(triplet: CommonLineTriplet) -> CommonLineTriplet:
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