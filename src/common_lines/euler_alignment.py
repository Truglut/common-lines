import numpy as np

from .alignment import BoolArray, CommonLineTriplet, FloatArray, _prepare_triplet
from .euler import EulerAngles

EulerPair = tuple[EulerAngles, EulerAngles]


def _alignment_geometry(
    triplet: CommonLineTriplet, eps: float
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """
    Return tilt-cosine numerator, denominator, and geometric validity.

    Input fields are already broadcast. Only nonparallel entries are divided
    when testing the cosine bounds; no input angles or estimates are replaced.
    """
    delta_1, delta_2, delta_3 = triplet.angular_differences()
    sin_delta_1, sin_delta_2 = np.sin(delta_1), np.sin(delta_2)

    # Dot products give cos(c) = cos(a)cos(b) + sin(a)sin(b)cos(tilt).
    numerator = np.asarray(np.cos(delta_3) - np.cos(delta_1) * np.cos(delta_2))
    denominator = np.asarray(sin_delta_1 * sin_delta_2)

    nonparallel = (np.abs(sin_delta_1) >= eps) & (np.abs(sin_delta_2) >= eps)
    cos_tilt = numerator[nonparallel] / denominator[nonparallel]
    valid = np.array(nonparallel, copy=True)

    # Preserve the Euler cosine tolerance, distinct from the matrix z² tolerance.
    valid[nonparallel] = np.abs(cos_tilt) <= 1.0 + eps

    return numerator, denominator, valid


def valid_euler_triplets(triplet: CommonLineTriplet, eps: float = 1.0e-6) -> BoolArray:
    """
    Identify triplets accepted by the Euler solver, without estimating poses.

    Return a boolean array of the six input angles' broadcast shape (shape ()
    for scalars). Real, finite angles are assumed. A triplet is valid when its
    line separations in images 1 and 2 are nonparallel and its implied tilt
    cosine lies within [-1 - eps, 1 + eps]. NumPy reports incompatible shapes.
    """
    _, _, valid = _alignment_geometry(_prepare_triplet(triplet), eps)
    return valid


def alignment_euler_angles(
    triplet: CommonLineTriplet, eps: float = 1.0e-6
) -> EulerPair:
    """
    Return direct and mirrored image-1-to-image-2 Euler angles in radians.

    All six input angles broadcast together; each Euler field has that shape.
    Negating the tilt gives the opposite handedness. ``eps`` bounds both
    near-parallel line separations (via their sines) and tolerated overshoot
    of the tilt cosine outside [-1, 1].

    Any geometrically invalid entry raises ValueError for the whole call.
    ``valid_euler_triplets`` uses the same geometric calculation for filtering.
    Real, finite input angles are assumed. Empty batches are accepted.
    """
    triplet = _prepare_triplet(triplet)
    numerator, denominator, valid = _alignment_geometry(triplet, eps)
    if not valid.all():
        raise ValueError("Degenerate or inconsistent common-line geometry.")

    tilt = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))

    quarter_turn = np.pi / 2
    direct = EulerAngles(
        rot=triplet.one_with_two.first_rad - quarter_turn,
        tilt=tilt,
        psi=quarter_turn - triplet.one_with_two.second_rad,
    )
    mirrored = EulerAngles(rot=direct.rot, tilt=-direct.tilt, psi=direct.psi)

    return direct, mirrored
