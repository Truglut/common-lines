import itertools

import numpy as np

from common_lines.geometry.euler import EulerAngles
from common_lines.types import BoolArray, FloatArray

from .triplets import CommonLineAngles, CommonLineTriplet, broadcast_triplet

GAUSSIAN_REFINEMENT_MAX_ITS = 100
GAUSSIAN_REFINEMENT_TOLERANCE = 1.0e-12


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
    """Return ZYZ in-plane angles from an oriented pair of common-line angles.

    Inputs are radians in image 1 and image 2. Return
    ``(angle_1 - pi/2, pi/2 - angle_2)`` without wrapping, using the convention
    of ``euler_zyz_to_matrix``. Scalars and NumPy arrays are accepted. The tilt
    remains undetermined by this pair and must be obtained from a third image.
    """
    quarter_turn = np.pi / 2

    return angle_1 - quarter_turn, quarter_turn - angle_2


def alignment_euler_angles(
    triplet: CommonLineTriplet, eps: float = 1.0e-6
) -> EulerAngles:
    """
    Return one arbitrary-handed image-1-to-image-2 Euler estimate in radians.

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
    return EulerAngles(rot=rot, tilt=tilt, psi=psi)


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

    detected_tilt = np.arccos(np.clip(numerator / denominator, -1.0, 1.0)).ravel()
    tilt, _ = _weighted_histogram_vote(
        detected_tilt=detected_tilt, bins=bins, sigma=sigma, weights=None
    )
    return tilt


def _weighted_histogram_vote(
    detected_tilt: np.ndarray,
    bins: int,
    sigma: float,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    if weights is None:
        weights = np.full_like(detected_tilt, fill_value=1.0)

    histogram_tilt = np.linspace(0.0, np.pi, bins)

    differences = (detected_tilt[:, np.newaxis] - histogram_tilt) / sigma
    histogram_height = np.sum(
        weights[:, None] * np.exp(-0.5 * differences**2),
        axis=0,
    )

    tilt = float(histogram_tilt[np.argmax(histogram_height)])

    # At a stationary point of a Gaussian vote sum, tilt equals the
    # Gaussian-weighted mean of the detected tilts. Normalization cancels.
    for _ in range(GAUSSIAN_REFINEMENT_MAX_ITS):
        log_gaussian_weights = -0.5 * ((detected_tilt - tilt) / sigma) ** 2
        gaussian_weights = np.exp(log_gaussian_weights - log_gaussian_weights.max())
        sum_weights = weights * gaussian_weights
        refined = float(np.sum(sum_weights * detected_tilt) / sum_weights.sum())

        change = abs(refined - tilt)
        tilt = refined
        if change <= GAUSSIAN_REFINEMENT_TOLERANCE:
            break

    confidence_gaussian_weights = np.exp(-0.5 * ((detected_tilt - tilt) / sigma) ** 2)
    confidence = float(np.sum(weights * confidence_gaussian_weights) / weights.sum())

    return tilt, confidence


def matrix_voting(
    angles: np.ndarray,
    valid: np.ndarray,
    voting_bins: int,
    voting_sigma: float,
    max_its: int,
    stopping_tol: float = 0.01,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """
    Calculate tilt angle of relative orientations with voting for a full array
    of common line angles, with optional voting weights

    Parameters
    ----------
    angles : np.ndarray
        Measured common-line angles between all image pairs. ``angles[i, j]`` is the
        angle of the common line between images ``i`` and ``j`` as measured in
        image ``i``.
    valid : np.ndarray
        Boolean mask indicating which third images are geometrically valid witnesses
        for common-line reconstitution. ``valid[i, j, k]`` is True if and only if
        image k is a valid third image witness for reconstitution of the ``(i, j)``
        pair.

    Returns
    -------
    np.ndarray
        ***
    """
    n_images = angles.shape[0]

    indices = np.arange(n_images)
    reliabilities = np.full((n_images, n_images), fill_value=1.0)
    reliabilities[indices, indices] = np.nan
    off_diagonal = np.ones_like(reliabilities, dtype=bool)
    off_diagonal[indices, indices] = False

    for _ in range(max_its):
        tilts, next_reliabilities = _voting_iteration(
            angles=angles,
            valid=valid,
            reliabilities=reliabilities,
            voting_bins=voting_bins,
            voting_sigma=voting_sigma,
            eps=eps,
        )

        max_change = np.max(
            np.abs(reliabilities[off_diagonal] - next_reliabilities[off_diagonal])
        )
        baseline = np.max(reliabilities[off_diagonal])
        reliabilities = next_reliabilities
        if max_change < stopping_tol * baseline:
            break

    return tilts


def _voting_iteration(
    angles: np.ndarray,
    valid: np.ndarray,
    reliabilities: np.ndarray,
    *,
    voting_bins: int,
    voting_sigma: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_images = angles.shape[0]
    indices = np.arange(n_images)
    out_tilts = np.empty((n_images, n_images), dtype=np.float64)
    out_tilts[indices, indices] = 0.0
    out_new_reliabilities = np.empty((n_images, n_images), dtype=np.float64)
    out_new_reliabilities[indices, indices] = np.nan

    def angles_at(i: int | np.ndarray, j: int | np.ndarray) -> CommonLineAngles:
        return CommonLineAngles(first_rad=angles[i, j], second_rad=angles[j, i])

    for i, j in itertools.combinations(range(n_images), 2):
        others = indices[valid[i, j]]
        triplet = CommonLineTriplet(
            one_with_two=angles_at(i, j),
            one_with_three=angles_at(i, others),
            two_with_three=angles_at(j, others),
        )
        num, den, _ = _alignment_geometry(triplet, eps=eps)
        detected_tilts = np.arccos(np.clip(num / den, -1.0, 1.0))

        weights = reliabilities[i, others] * reliabilities[j, others]

        tilt, confidence = _weighted_histogram_vote(
            detected_tilt=detected_tilts,
            bins=voting_bins,
            sigma=voting_sigma,
            weights=weights,
        )

        out_new_reliabilities[i, j] = confidence
        out_new_reliabilities[j, i] = confidence
        out_tilts[i, j] = tilt

        # Euler angles of the transpose/inverse are rot=-psi, tilt=-tilt, psi=-rot;
        # rot, -tilt, psi <==> rot + pi, tilt, psi + pi
        out_tilts[j, i] = tilt

    return out_tilts, out_new_reliabilities
