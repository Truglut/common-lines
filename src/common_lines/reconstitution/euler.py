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
    """
    Estimate one consensus tilt from all broadcast triplets.

    Each tilt casts a Gaussian vote on the interval [0, pi].
    A grid of ``bins`` points locates a peak, then Gaussian
    mean shift refines it to avoid bin quantization.

    Parameters
    ----------
    triplet : CommonLineTriplet
        Detected common line angles. All six angle fields broadcast together;
        each broadcast entry contributes one vote.
    bins : int
        Number of histogram bins used to initially locate the peak.
    sigma : float
        Positive, finite bandwidth of Gaussian votes, in radians.
    eps : float, optional
        Numerical tolerance for geometrically degenerate configurations, by default 1.0e-6

    Returns
    -------
    float
        Consensus tilt angle for the relative orientation of image 2 with respect to
        image 1 in the triplet.

    Raises
    ------
    ValueError
        If any of the provided triplets are geometrically degenerate, or if an
        empty triplet object is provided.
    """
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
    detected_tilt: FloatArray,
    bins: int,
    sigma: float,
    weights: FloatArray | None = None,
) -> tuple[float, float]:
    """
    Return a Gaussian consensus tilt and normalized support at that tilt.

    Parameters
    ----------
    detected_tilt : FloatArray
        Nonempty one-dimensional array of finite angles in [0, pi], in radians.
    bins : int
        Number of histogram bins used to locate the initial peak.
        Must be an integer >= 2
    sigma : float
        Positive, finite bandwidth of the Gaussian votes, in radians.
    weights : FloatArray | None, optional
        Finite, nonnegative reliability weights with the same shape as
        ``detected_tilt``, by default None (equal weights).

    Returns
    -------
    float
        Consensus angle derived from the Gaussian voting.
    float
        Confidence value measuring agreement between all provided estimates.
        Computed as sum(w * exp(-0.5 * ((t - tilt) / sigma)**2)) / sum(w).
        A single positive-weight vote has confidence one.

    Raises
    ------
    ValueError
        If ``detected_tilt`` is empty or all provided weights are non-positive.

    Notes
    -----
    Shapes, finite values, angle bounds, bins, and sigma are assumed valid.
    Angles with non-positive weights are ignored. The interval [0, pi] is nonperiodic.
    The grid initializes Gaussian mean shift, which returns a refined estimate
    after convergence or the iteration limit; a global maximum is not guaranteed.
    Scaling all weights by a positive constant preserves both outputs.
    """
    if weights is None:
        weights = np.ones_like(detected_tilt)

    positive = weights > 0
    if not np.any(positive):
        raise ValueError("At least one weight must be positive.")
    # Zero-weight witnesses must not determine the refinement's exponent shift.
    detected_tilt = detected_tilt[positive]
    weights = weights[positive]

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
        gaussian_weights = np.exp(log_gaussian_weights - np.max(log_gaussian_weights))
        sum_weights = weights * gaussian_weights
        refined = float(np.sum(sum_weights * detected_tilt) / sum_weights.sum())

        change = abs(refined - tilt)
        tilt = refined
        if change <= GAUSSIAN_REFINEMENT_TOLERANCE:
            break

    confidence_gaussian_weights = np.exp(-0.5 * ((detected_tilt - tilt) / sigma) ** 2)
    confidence = float(np.sum(weights * confidence_gaussian_weights) / weights.sum())

    return tilt, confidence


def soft_vote_tilts(
    angle_1: FloatArray,
    angle_2: FloatArray,
    voting_bins: int,
    voting_sigma: float,
    max_its: int,
    stopping_tol: float = 0.01,
    eps: float = 1.0e-6,
) -> FloatArray:
    """
    Estimate a symmetric matrix of relative tilts by iterative soft voting.

    Off-diagonal reliabilities start at one. Each witness k for pair (i, j)
    receives weight r[i, k] * r[j, k] from the preceding iteration. Gaussian
    consensus support supplies the next reliability. Updates are synchronous.

    Parameters
    ----------
    angle_1, angle_2 : FloatArray
        Square (n, n) arrays, with n >= 3, of oriented common-line angles in
        radians. ``angle_1[i, j]`` is measured in image i and ``angle_2[i, j]``
        describes the same oriented line in image j. Detector-reversal
        corrections must already be included; angle_2 need not be angle_1.T.
        Off-diagonal entries must be real and finite; diagonals are unused.
        Shapes and finite values are assumed, not checked here.
    voting_bins : int
        Number of inclusive grid points on [0, pi]; must be at least two.
    voting_sigma : float
        Positive, finite Gaussian bandwidth in radians.
    max_its : int
        Maximum number of synchronous updates; must be at least one.
    stopping_tol : float, optional
        Nonnegative, finite relative reliability tolerance. Stop when the
        largest off-diagonal change is strictly below this tolerance times
        the previous largest reliability. Zero disables early stopping.
    eps : float, optional
        Geometric tolerance, as in ``alignment_euler_angles``.

    Returns
    -------
    FloatArray
        Symmetric (n, n) tilt matrix in radians, with zero diagonal and
        off-diagonal values in [0, pi], from the final performed update.

    Raises
    ------
    ValueError
        If max_its is non-positive or any pair has no geometrically valid
        positive-weight witness. Unsupported-pair errors identify the pair.

    Notes
    -----
    This method is based on the following work:
    Wang, X., Xu, Y., Zheng, H. et al.
    Soft-voting for robust common line assessment in cryo-EM angular reconstruction.
    BMC Bioinformatics (2026). https://doi.org/10.1186/s12859-026-06721-8
    """
    n_images = angle_1.shape[0]

    if max_its <= 0:
        raise ValueError("max_its must be a positive integer")

    indices = np.arange(n_images)
    reliabilities = np.full((n_images, n_images), fill_value=1.0)
    reliabilities[indices, indices] = np.nan
    off_diagonal = np.ones_like(reliabilities, dtype=bool)
    off_diagonal[indices, indices] = False

    for _ in range(max_its):
        tilts, next_reliabilities = _voting_iteration(
            angle_1=angle_1,
            angle_2=angle_2,
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
    angle_1: FloatArray,
    angle_2: FloatArray,
    reliabilities: FloatArray,
    *,
    voting_bins: int,
    voting_sigma: float,
    eps: float,
) -> tuple[FloatArray, FloatArray]:
    """
    Run one synchronous iteration of soft-voting common-line reconstitution.

    angle_1 and angle_2 follow the input conventions of ``soft_vote_tilts``.
    reliabilities is a symmetric (n, n) array with finite off-diagonal entries
    in [0, 1]; its diagonal is unused and may be NaN. These are preconditions,
    not validated here. Voting parameters and eps have the same meanings as
    in ``soft_vote_tilts``.

    For each unordered pair, all other images are considered as witnesses.
    Geometry and tilt candidates are calculated together, retaining only
    geometrically valid witnesses with positive incident-reliability products.
    No validity cube is stored. A pair without support raises ValueError
    identifying its indices.

    Return freshly allocated tilt and confidence matrices. Both are symmetric;
    tilt has zero diagonal, and confidence has NaN diagonal. Confidence is the
    normalized Gaussian support defined by ``_weighted_histogram_vote`` and
    supplies the next reliability. Inputs are not modified. Only old
    reliabilities are read, so pair traversal order does not change the update.
    """
    n_images = angle_1.shape[0]
    indices = np.arange(n_images)
    out_tilts = np.empty((n_images, n_images), dtype=np.float64)
    out_tilts[indices, indices] = 0.0
    out_new_reliabilities = np.empty((n_images, n_images), dtype=np.float64)
    out_new_reliabilities[indices, indices] = np.nan

    def angles_at(i: int | np.ndarray, j: int | np.ndarray) -> CommonLineAngles:
        return CommonLineAngles(first_rad=angle_1[i, j], second_rad=angle_2[i, j])

    available_mask = np.ones(n_images, dtype=bool)
    for i, j in itertools.combinations(range(n_images), 2):
        available_mask[i] = False
        available_mask[j] = False

        others = indices[available_mask]

        triplet = CommonLineTriplet(
            one_with_two=angles_at(i, j),
            one_with_three=angles_at(i, others),
            two_with_three=angles_at(j, others),
        )
        num, den, valid = _alignment_geometry(broadcast_triplet(triplet), eps=eps)

        weights = reliabilities[i, others] * reliabilities[j, others]
        support = valid & (weights > 0)

        if not support.any():
            raise ValueError(f"Unsupported image pair {(i, j)}.")

        detected_tilts = np.arccos(np.clip(num[support] / den[support], -1.0, 1.0))

        tilt, confidence = _weighted_histogram_vote(
            detected_tilt=detected_tilts,
            weights=weights[support],
            bins=voting_bins,
            sigma=voting_sigma,
        )

        out_new_reliabilities[i, j] = confidence
        out_new_reliabilities[j, i] = confidence
        out_tilts[i, j] = tilt

        # Euler angles of the transpose/inverse are rot=-psi, tilt=-tilt, psi=-rot;
        # rot, -tilt, psi <==> rot + pi, tilt, psi + pi
        out_tilts[j, i] = tilt

        available_mask[i] = True
        available_mask[j] = True

    return out_tilts, out_new_reliabilities
