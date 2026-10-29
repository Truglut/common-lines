import math

from .alignment import CommonLineTriplet
from .euler import EulerAngles


def _alignment_tilt(triplet: CommonLineTriplet, eps: float) -> float:
    delta_1, delta_2, delta_3 = triplet.angular_differences()
    if not all(math.isfinite(angle) for angle in (delta_1, delta_2, delta_3)):
        raise ValueError("Common-line separations must be finite.")

    sin_delta_1, sin_delta_2 = math.sin(delta_1), math.sin(delta_2)
    if abs(sin_delta_1) < eps or abs(sin_delta_2) < eps:
        raise ValueError("Degenerate configuration: common lines are parallel.")

    # Dot products of the three common lines give
    # cos(c) = cos(a)*cos(b) + sin(a)*sin(b)*cos(tilt).
    cos_tilt = (math.cos(delta_3) - math.cos(delta_1) * math.cos(delta_2)) / (
        sin_delta_1 * sin_delta_2
    )
    if abs(cos_tilt) > 1.0 + eps:
        raise ValueError("Inconsistent common-line angles: no valid 3D configuration.")

    return math.acos(max(-1.0, min(1.0, cos_tilt)))


def alignment_euler_angles(
    triplet: CommonLineTriplet, eps: float = 1.0e-6
) -> tuple[EulerAngles, EulerAngles]:
    """Return direct and mirrored image-1-to-image-2 orientations in radians.

    The first solution fixes common line 2-3 to have positive reference z;
    negating the tilt gives the mirrored solution.
    ``eps`` bounds both near-parallel line separations (via their sines) and
    tolerated overshoot of the tilt cosine outside [-1, 1].
    """
    tilt = _alignment_tilt(triplet, eps)

    quarter_turn = math.pi / 2
    direct = EulerAngles(
        rot=triplet.one_with_two.first_rad - quarter_turn,
        tilt=tilt,
        psi=quarter_turn - triplet.one_with_two.second_rad,
    )
    mirrored = EulerAngles(
        rot=direct.rot,
        tilt=-tilt,
        psi=direct.psi,
    )

    return direct, mirrored
