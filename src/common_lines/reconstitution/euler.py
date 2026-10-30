import numpy as np

from ..euler import EulerAngles
from .triplets import BoolArray, CommonLineTriplet, FloatArray, broadcast_triplet

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
    _, _, valid = _alignment_geometry(broadcast_triplet(triplet), eps)
    return valid


def rot_psi_from_angle_pair(
    angle_1: FloatArray | float, angle_2: FloatArray | float
) -> tuple[FloatArray | float, FloatArray | float]:
    quarter_turn = np.pi / 2

    return angle_1 - quarter_turn, quarter_turn - angle_2


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
    triplet = broadcast_triplet(triplet)
    numerator, denominator, valid = _alignment_geometry(triplet, eps)
    if not valid.all():
        raise ValueError("Degenerate or inconsistent common-line geometry.")

    tilt = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))

    rot, psi = rot_psi_from_angle_pair(
        triplet.one_with_two.first_rad, triplet.one_with_two.second_rad
    )
    direct = EulerAngles(rot=rot, tilt=tilt, psi=psi)
    mirrored = EulerAngles(rot=direct.rot, tilt=-direct.tilt, psi=direct.psi)

    return direct, mirrored


def vote_tilt(
    triplet: CommonLineTriplet, bins: int, sigma: float, eps: float = 1.0e-6
) -> float:
    """Estimate one consensus tilt from all broadcast triplets.

    Each tilt casts a Gaussian vote on the interval [0, pi], with bandwidth
    ``sigma`` in radians. The endpoints are distinct, not periodic. A grid of
    ``bins`` points locates a peak, then Gaussian mean shift refines it without
    bin quantization. Thus a single vote retains its original tilt.

    Invalid geometry or an empty batch raises ValueError.
    """
    if not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer of at least 2.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite, in radians.")

    numerator, denominator, valid = _alignment_geometry(broadcast_triplet(triplet), eps)
    if not valid.all():
        raise ValueError("Degenerate or inconsistent common-line geometry.")
    if valid.size == 0:
        raise ValueError("At least one triplet is required for voting.")

    histogram_tilt = np.linspace(0.0, np.pi, bins)
    detected_tilt = np.arccos(np.clip(numerator / denominator, -1.0, 1.0)).ravel()
    differences = (detected_tilt[:, np.newaxis] - histogram_tilt) / sigma
    histogram_height = np.exp(-0.5 * differences**2).sum(axis=0)
    tilt = float(histogram_tilt[np.argmax(histogram_height)])

    # At a stationary point of a Gaussian vote sum, tilt equals the
    # Gaussian-weighted mean of the detected tilts. Normalization cancels.
    for _ in range(100):
        log_weights = -0.5 * ((detected_tilt - tilt) / sigma) ** 2
        weights = np.exp(log_weights - log_weights.max())
        refined = float(np.sum(weights * detected_tilt) / weights.sum())
        if abs(refined - tilt) <= 1.0e-12:
            return refined
        tilt = refined
    return tilt
