from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True, slots=True)
class CommonLineResult:
    """
    Best common-line matches for one or more explicit sinogram pairs.

    All fields are one-dimensional arrays with one entry per compared pair.
    A single pair is represented by arrays of length one, which keeps the
    output format identical for scalar and batched calls.

    Attributes
    ----------
    index_1, index_2
        Selected angular-profile indices in the first and second sinograms.
        Because ``images_1[i]`` is only compared to ``images_2[i]``, ``index_1[i]``
        contains the angular index of their common line in ``images_1[i]``, and
        ``index_2[i]`` contains the index of the common line in ``images_2[i]``.
    correlation
        Pearson correlation of each selected profile pair.
    reversed_match
        Boolean array identifying matches that used the reversed detector
        direction of the second profile. It is ``None`` when reversed profiles
        were not considered.
    """

    index_1: np.ndarray
    index_2: np.ndarray
    correlation: np.ndarray
    reversed_match: np.ndarray | None = None

    def oriented_angles(
        self, theta_1: np.ndarray, theta_2: np.ndarray, radians: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert the stored indices into oriented angular matrices.

        Parameters
        ----------
        theta_1
            Angular grid used to generate the first batch of sinograms, with shape ``(A1,)``.
        theta_2
            Angular grid used to generate the second batch of sinograms, with shape ``(A2,)``.
        radians
            If True, the input angular grids and the outputs are in radians. Otherwise
            all angles are in degrees.

        Returns
        -------
        angle_1, angle_2
            Arrays with shape ``(n_images, )``. For reversed matches,
            180 degrees is added to ``angle_2`` modulo 360 degrees.
        """
        theta_1 = np.asarray(theta_1)
        theta_2 = np.asarray(theta_2)

        angle_1 = theta_1[self.index_1]
        angle_2 = theta_2[self.index_2]

        if self.reversed_match is not None:
            full_turn = 2 * np.pi if radians else 360.0
            angle_2 = np.mod(
                angle_2 + 0.5 * full_turn * self.reversed_match,
                full_turn,
            )

        return angle_1, angle_2


@dataclass(frozen=True, slots=True)
class CommonLineGraph:
    """
    Common-line results for all image pairs.

    The stored arrays have shape ``(n_images, n_images)``.

    Attributes
    ----------
    index
        ``index[i, j]`` is the angular-profile index of the common line
        with image ``j``, expressed in image ``i``. Therefore, the two angular
        indices associated with pair ``(i, j)`` are ``index[i, j]`` and
        ``index[j, i]``.
    correlation
        Correlation value of each selected match.
    reversed_match
        Boolean matrix identifying detector-reversed matches. It is ``None``
        when reversed profiles were not considered.
    n_angles
        Number of angular profiles in each sinogram.
    """

    index: np.ndarray
    correlation: np.ndarray
    reversed_match: np.ndarray | None = None
    n_angles: int = 0

    def oriented_angles(
        self,
        theta: np.ndarray,
        radians: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert the stored indices into oriented angular matrices.

        For pair ``(i, j)``, the returned angles correspond to the common
        line expressed in images ``i`` and ``j``, respectively.
        """
        theta = np.asarray(theta)
        _validate_theta(theta, self.n_angles, "theta")

        angle_1 = theta[self.index]
        angle_2 = theta[self.index.T]

        if self.reversed_match is not None:
            full_turn = 2 * np.pi if radians else 360.0
            angle_2 = np.mod(
                angle_2 + 0.5 * full_turn * self.reversed_match,
                full_turn,
            )

        return angle_1, angle_2


class _MatchResult(NamedTuple):
    """Best angular indices and correlations for one set of matches."""

    index_1: np.ndarray
    index_2: np.ndarray
    correlation: np.ndarray


def detect_common_lines(
    sinograms_1: np.ndarray,
    sinograms_2: np.ndarray,
    *,
    eps: float = 1.0e-8,
    allow_reversed: bool = False,
) -> CommonLineResult:
    """
    Detect common lines between corresponding pairs of sinograms.

    A single sinogram may be compared with another single sinogram or
    broadcast against a batch. When both inputs are batches, element ``b`` of
    the first batch is compared only with element ``b`` of the second batch.

    Parameters
    ----------
    sinograms_1
        First sinogram or batch, with shape ``(P, A1)`` or ``(B, P, A1)``.
    sinograms_2
        Second sinogram or batch, with shape ``(P, A2)`` or ``(B, P, A2)``.
    eps
        Columns with centered norm at most ``eps`` are treated as zero.
    allow_reversed
        Also compare profiles after reversing the detector axis of the second
        sinogram.

    Returns
    -------
    CommonLineResult
        Best match for each corresponding sinogram pair.
    """
    # Input validation
    _validate_eps(eps)

    s1 = _as_batch_sinograms(sinograms_1)
    s2 = _as_batch_sinograms(sinograms_2)
    s1, s2 = _broadcast_batches(s1, s2)

    if s1.shape[1] != s2.shape[1]:
        raise ValueError(
            "sinograms_1 and sinograms_2 must have the same detector "
            f"dimension. Got {s1.shape[1]} and {s2.shape[1]}."
        )

    # Normalize sinogram columns to easily compute correlation as dot product
    normalized_1 = _normalize_columns(s1, eps=eps)
    normalized_2 = _normalize_columns(s2, eps=eps)

    # Detect best pairwise matches between sinogram columns
    normal = _detect_pairwise_matches(normalized_1, normalized_2)
    selected = normal
    reversed_selected = None

    # Perform detection with reversed detector and store results
    if allow_reversed:
        reversed_candidate = _detect_pairwise_matches(
            normalized_1,
            normalized_2[:, ::-1, :],
        )
        selected, reversed_selected = _choose_best_matches(normal, reversed_candidate)

    return CommonLineResult(
        index_1=selected.index_1,
        index_2=selected.index_2,
        correlation=selected.correlation,
        reversed_match=reversed_selected,
    )


def detect_common_lines_all_pairs(
    sinograms: np.ndarray,
    *,
    eps: float = 1.0e-8,
    allow_reversed: bool = False,
    block_size: int = 4,
) -> CommonLineGraph:
    """
    Detect common lines between all sinogram pairs.

    Parameters
    ----------
    sinograms
        Sinogram batch with shape ``(n, detector_size, n_angles)``.
    eps
        Columns with centered norm at most ``eps`` are treated as zero.
    allow_reversed
        Also compare profiles after reversing the detector axis of the second
        sinogram.
    block_size
        Number of images processed along each block dimension. Temporary
        correlation memory scales approximately as ``block_size**2 * n_angles**2``.

    Returns
    -------
    CommonLineGraph
        Compact arrays with shape ``(n_images, n_images)`` containing all directed
        pairwise match indices. Use ``CommonLineGraph.oriented_angles`` with the
        angular grid when angle matrices are required.
    """
    _validate_eps(eps)
    block_size = _validate_block_size(block_size)

    # Prepare features: sinograms with normalized columns in original and reversed order
    sinograms = _as_sinogram_batch(sinograms)
    n_images, _, n_angles = sinograms.shape
    features, reversed_features = _prepare_common_line_features(
        sinograms,
        eps=eps,
        include_reversed=allow_reversed,
    )

    # Prepare output arrays
    index_dtype = _angular_index_dtype(n_angles)
    index = np.empty((n_images, n_images), dtype=index_dtype)
    correlation = np.empty((n_images, n_images), dtype=features.dtype)
    reversed_match = (
        np.empty((n_images, n_images), dtype=bool) if allow_reversed else None
    )

    # Process features in blocks to detect best matches
    for i_start in range(0, n_images, block_size):
        i_stop = min(i_start + block_size, n_images)
        i_slice = slice(i_start, i_stop)
        left = features[i_slice]

        for j_start in range(i_start, n_images, block_size):
            j_stop = min(j_start + block_size, n_images)
            j_slice = slice(j_start, j_stop)
            right = features[j_slice]

            selected = _detect_block_matches(left, right)
            block_reversed = None

            if reversed_features is not None:
                # Detect matches also for the reversed features and keep best match
                right_reversed = reversed_features[j_start:j_stop]
                reversed_candidate = _detect_block_matches(left, right_reversed)
                selected, block_reversed = _choose_best_matches(
                    selected, reversed_candidate
                )

            same_block = i_start == j_start  # j_start starts from i_start

            if same_block:
                # When comparing a block against itself, we enforce the graph's
                # symmetry by doing  index[i, j] = selected.index_1[i, j] and
                # index[j, i] = selected.index_2[i, j]
                n_block = i_stop - i_start
                row, col = np.triu_indices(n_block, k=1)

                global_i = i_start + row
                global_j = i_start + col  # i_start == j_start here

                index[global_i, global_j] = selected.index_1[row, col]
                index[global_j, global_i] = selected.index_2[row, col]

                correlation[global_i, global_j] = selected.correlation[row, col]
                correlation[global_j, global_i] = selected.correlation[row, col]

                if reversed_match is not None:
                    assert block_reversed is not None
                    reversed_match[global_i, global_j] = block_reversed[row, col]
                    reversed_match[global_j, global_i] = block_reversed[row, col]

            else:
                index[i_slice, j_slice] = selected.index_1
                index[j_slice, i_slice] = selected.index_2.T

                correlation[i_slice, j_slice] = selected.correlation
                correlation[j_slice, i_slice] = selected.correlation.T

                if reversed_match is not None:
                    assert block_reversed is not None
                    reversed_match[i_slice, j_slice] = block_reversed
                    reversed_match[j_slice, i_slice] = block_reversed.T

    # The diagonal does not have a defined meaning for common lines, initialize it
    # for consistency
    diag = np.arange(n_images)
    index[diag, diag] = 0
    correlation[diag, diag] = np.nan
    if reversed_match is not None:
        reversed_match[diag, diag] = False

    return CommonLineGraph(
        index=index,
        correlation=correlation,
        reversed_match=reversed_match,
        n_angles=n_angles,
    )


def _detect_pairwise_matches(
    normalized_1: np.ndarray,
    normalized_2: np.ndarray,
) -> _MatchResult:
    """Compute and reduce correlation matrices for an explicit pair batch."""
    correlation = np.einsum(
        "bpa,bpc->bac",
        normalized_1,
        normalized_2,
        optimize=True,
    )
    return _reduce_pairwise_correlations(correlation)


def _detect_block_matches(
    normalized_left: np.ndarray,
    normalized_right: np.ndarray,
) -> _MatchResult:
    """
    Calculates the correlation between each element of the two arrays (assuming their
    first axis is a batch dimension) and returns the indices of the best matches,
    along with their correlation values

    Parameters
    ----------
    normalized_left : np.ndarray
        Features array of shape ``(n_left, n_angles_left, detector_size)`` whose
        last dimension is normalized, i.e. ``normalized_left[i, j, :]`` has norm 1
        for every i, j.
    normalized_right : np.ndarray
        Features array of shape ``(n_right, n_angles_right, detector_size)``, also
        normalized.

    Returns
    -------
    _MatchResult
        Best correlation matches along with their values.
    """
    n_left, n_angles_left, detector_size = normalized_left.shape
    n_right, n_angles_right, _ = normalized_right.shape

    left_2d = normalized_left.reshape(-1, detector_size)  # (..., detector_size)
    right_2d = normalized_right.reshape(-1, detector_size)  # (..., detector_size)

    correlation = (
        (left_2d @ right_2d.T)  # (..., detector_size) @ (detector_size, ...)
        .reshape(n_left, n_angles_left, n_right, n_angles_right)
        .transpose(0, 2, 1, 3)
        .reshape(n_left, n_right, -1)
    )

    indices_flat = correlation.argmax(axis=-1)
    corr_values = np.take_along_axis(
        correlation, indices_flat[..., None], axis=-1
    ).squeeze(-1)

    index_left, index_right = np.unravel_index(
        indices_flat, shape=(n_angles_left, n_angles_right)
    )

    return _MatchResult(
        index_1=index_left, index_2=index_right, correlation=corr_values
    )


def _reduce_pairwise_correlations(correlation: np.ndarray) -> _MatchResult:
    """Extract the best angular pair from each pairwise correlation matrix."""
    batch_size, _, n_angles_2 = correlation.shape
    flattened = correlation.reshape(batch_size, -1)
    flat_index = np.argmax(flattened, axis=1)

    index_1 = flat_index // n_angles_2
    index_2 = flat_index % n_angles_2
    score = np.take_along_axis(
        flattened,
        flat_index[:, None],
        axis=1,
    )[:, 0]

    return _MatchResult(index_1, index_2, score)


def _choose_best_matches(
    normal: _MatchResult, reversed_match: _MatchResult
) -> tuple[_MatchResult, np.ndarray]:
    """
    Select the better normal or detector-reversed match.

    Ties are resolved in favor of the normal match.
    """
    use_reversed = reversed_match.correlation > normal.correlation

    selected = _MatchResult(
        index_1=np.where(use_reversed, reversed_match.index_1, normal.index_1),
        index_2=np.where(use_reversed, reversed_match.index_2, normal.index_2),
        correlation=np.where(
            use_reversed,
            reversed_match.correlation,
            normal.correlation,
        ),
    )

    return selected, use_reversed


def _prepare_common_line_features(
    sinograms: np.ndarray,
    *,
    eps: float,
    include_reversed: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize columns and arrange angular profiles as contiguous rows."""
    normalized = _normalize_columns(sinograms, eps=eps)
    features = np.ascontiguousarray(normalized.transpose(0, 2, 1))

    reversed_features = None
    if include_reversed:
        reversed_features = np.ascontiguousarray(features[..., ::-1])

    return features, reversed_features


def _as_batch_sinograms(sinograms: np.ndarray) -> np.ndarray:
    """Convert a single sinogram to a batch and ensure floating-point data."""
    sinograms = np.asarray(sinograms)

    if sinograms.ndim == 2:
        sinograms = sinograms[None, ...]
    elif sinograms.ndim != 3:
        raise ValueError(
            "sinograms must have shape (P, A) or (B, P, A). "
            f"Got array with shape {sinograms.shape}."
        )

    return _validate_and_convert_sinograms(sinograms)


def _as_sinogram_batch(sinograms: np.ndarray) -> np.ndarray:
    """Validate an explicit sinogram batch with shape ``(N, P, A)``."""
    sinograms = np.asarray(sinograms)

    if sinograms.ndim != 3:
        raise ValueError(
            "sinograms must have shape (N, P, A). "
            f"Got array with shape {sinograms.shape}."
        )

    return _validate_and_convert_sinograms(sinograms)


def _validate_and_convert_sinograms(sinograms: np.ndarray) -> np.ndarray:
    """Reject empty dimensions and convert non-floating inputs to float64."""
    if any(size == 0 for size in sinograms.shape):
        raise ValueError("sinograms must not contain empty dimensions.")

    if np.iscomplexobj(sinograms):
        raise TypeError("sinograms must contain real-valued data.")

    if not np.issubdtype(sinograms.dtype, np.floating):
        sinograms = sinograms.astype(np.float64, copy=False)

    return sinograms


def _broadcast_batches(
    sinograms_1: np.ndarray,
    sinograms_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast two sinogram batches when one has length one."""
    batch_1 = sinograms_1.shape[0]
    batch_2 = sinograms_2.shape[0]

    if batch_1 == batch_2:
        return sinograms_1, sinograms_2
    if batch_1 == 1:
        return (
            np.broadcast_to(sinograms_1, (batch_2, *sinograms_1.shape[1:])),
            sinograms_2,
        )
    if batch_2 == 1:
        return (
            sinograms_1,
            np.broadcast_to(sinograms_2, (batch_1, *sinograms_2.shape[1:])),
        )

    raise ValueError(
        "Batch dimensions must match unless one input contains a single "
        f"sinogram. Got batch sizes {batch_1} and {batch_2}."
    )


def _normalize_columns(
    sinograms: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    """Center and normalize every angular profile along the detector axis."""
    centered = sinograms - sinograms.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)

    normalized = np.zeros_like(centered)
    np.divide(centered, norms, out=normalized, where=norms > eps)
    return normalized


def _angular_index_dtype(n_angles: int) -> np.dtype:
    """Return the smallest unsigned dtype that stores all angular indices."""
    if n_angles < 1:
        raise ValueError("n_angles must be at least 1.")
    return np.dtype(np.min_scalar_type(n_angles - 1))


def _validate_theta(theta: np.ndarray, n_angles: int, name: str) -> None:
    """Check that an angle grid has the expected one-dimensional shape."""
    theta = np.asarray(theta)
    if theta.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if theta.size != n_angles:
        raise ValueError(
            f"{name} must have exactly {n_angles} entries. Got {theta.size}."
        )


def _validate_block_size(block_size: int) -> int:
    """Validate and normalize the image block size."""
    if not isinstance(block_size, (int, np.integer)) or isinstance(block_size, bool):
        raise TypeError("block_size must be an integer.")
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    return int(block_size)


def _validate_eps(eps: float) -> None:
    """Validate the normalization tolerance."""
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite non-negative value.")
