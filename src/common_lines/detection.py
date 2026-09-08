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
        self, theta_1: np.ndarray, theta_2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert the stored indices into oriented angular matrices.

        Parameters
        ----------
        theta_1
            Angular grid used to generate the first batch of sinograms, with shape ``(A1,)``.
        theta_2
            Angular grid used to generate the second batch of sinograms, with shape ``(A2,)``.

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
            angle_2 = np.mod(
                angle_2 + 180.0 * self.reversed_match,
                360.0,
            )

        return angle_1, angle_2


@dataclass(frozen=True, slots=True)
class CommonLineGraph:
    """
    Common-line results for all directed image pairs.

    The stored arrays have shape ``(n_images, n_images)``. Angular values are
    deliberately not stored because they can be reconstructed cheaply from
    the indices and the angular grid. This avoids retaining two additional
    floating-point matrices when only the indices or correlations are needed.

    Attributes
    ----------
    index_1, index_2
        Selected angular-profile indices in the first and second image.
    correlation
        Correlation value of each selected match.
    reversed_match
        Boolean matrix identifying detector-reversed matches. It is ``None``
        when reversed profiles were not considered.
    n_angles: int, optional
        Length of the angular array used to compute the sinograms.
        Default is 0 for an empty object.
    """

    index_1: np.ndarray
    index_2: np.ndarray
    correlation: np.ndarray
    reversed_match: np.ndarray | None = None
    n_angles: int = 0

    def oriented_angles(
        self,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert the stored indices into oriented angular matrices.

        Parameters
        ----------
        theta
            Angular grid used to generate the sinograms, with shape ``(A,)``.

        Returns
        -------
        angle_1, angle_2
            Arrays with shape ``(n_images, n_images)``. For reversed matches,
            180 degrees is added to ``angle_2`` modulo 360 degrees.
        """
        theta = np.asarray(theta)
        _validate_theta(theta, self.n_angles, "theta")

        angle_1 = theta[self.index_1]
        angle_2 = theta[self.index_2]

        if self.reversed_match is not None:
            angle_2 = np.mod(
                angle_2 + 180.0 * self.reversed_match,
                360.0,
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
    theta_1: np.ndarray | None = None,
    theta_2: np.ndarray | None = None,
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
    theta_1, theta_2
        Optional angle grids with shapes ``(A1,)`` and ``(A2,)``.
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

    n_angles_1 = s1.shape[2]
    n_angles_2 = s2.shape[2]
    _validate_theta(theta_1, n_angles_1, "theta_1")
    _validate_theta(theta_2, n_angles_2, "theta_2")

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
        selected, reversed_selected = _choose_best_matches(
            normal,
            reversed_candidate,
            n_angles_2=n_angles_2,
        )

    return CommonLineResult(
        index_1=selected.index_1,
        index_2=selected.index_2,
        correlation=selected.correlation,
        reversed_match=reversed_selected,
    )


def detect_common_lines_all_pairs_blocked(
    sinograms: np.ndarray,
    *,
    eps: float = 1.0e-8,
    allow_reversed: bool = False,
    block_size: int = 4,
) -> CommonLineGraph:
    """Detect common lines between all sinogram pairs using image blocks.

    All columns are normalized once. Image blocks are flattened into matrices
    of angular profiles, so each block-to-block comparison is evaluated by one
    matrix multiplication. Every correlation block is reduced immediately;
    the full tensor with shape ``(N, N, A, A)`` is never allocated.

    Only the upper triangle of off-diagonal image blocks is computed. Results
    for the opposite directed pairs are obtained by transposing the image-pair
    axes and exchanging the two angular indices.

    Parameters
    ----------
    sinograms
        Sinogram batch with shape ``(N, P, A)``.
    eps
        Columns with centered norm at most ``eps`` are treated as zero.
    allow_reversed
        Also compare profiles after reversing the detector axis of the second
        sinogram.
    block_size
        Number of images processed along each block dimension. Temporary
        correlation memory scales approximately as ``block_size**2 * A**2``.

    Returns
    -------
    CommonLineGraph
        Compact arrays with shape ``(N, N)`` containing all directed pairwise
        match indices. Use ``CommonLineGraph.oriented_angles`` with the angular
        grid when angle matrices are required.
    """
    _validate_eps(eps)
    block_size = _validate_block_size(block_size)

    # Prepare features: sinograms with normalized columns in original and reversed order
    sinograms = _as_sinogram_batch(sinograms)
    n_images, detector_size, n_angles = sinograms.shape
    features, reversed_features = _prepare_common_line_features(
        sinograms,
        eps=eps,
        include_reversed=allow_reversed,
    )

    # Prepare output arrays
    index_dtype = _angular_index_dtype(n_angles)
    index_1 = np.empty((n_images, n_images), dtype=index_dtype)
    index_2 = np.empty_like(index_1)
    correlation = np.empty((n_images, n_images), dtype=features.dtype)
    reversed_match = (
        np.empty((n_images, n_images), dtype=bool) if allow_reversed else None
    )

    # Process features in blocks to detect best matches
    for i_start in range(0, n_images, block_size):
        i_stop = min(i_start + block_size, n_images)
        n_left = i_stop - i_start
        left = features[i_start:i_stop].reshape(
            n_left * n_angles,
            detector_size,
        )

        for j_start in range(i_start, n_images, block_size):
            j_stop = min(j_start + block_size, n_images)
            n_right = j_stop - j_start
            right = features[j_start:j_stop].reshape(
                n_right * n_angles,
                detector_size,
            )

            selected = _detect_block_matches(
                left,
                right,
                n_left=n_left,
                n_right=n_right,
                n_angles=n_angles,
            )
            block_reversed = None

            if reversed_features is not None:
                # Detect matches also for the reversed features and keep best match
                right_reversed = reversed_features[j_start:j_stop].reshape(
                    n_right * n_angles,
                    detector_size,
                )
                reversed_candidate = _detect_block_matches(
                    left,
                    right_reversed,
                    n_left=n_left,
                    n_right=n_right,
                    n_angles=n_angles,
                )
                selected, block_reversed = _choose_best_matches(
                    selected,
                    reversed_candidate,
                    n_angles_2=n_angles,
                )

            # Store results in output arrays
            i_slice = slice(i_start, i_stop)
            j_slice = slice(j_start, j_stop)
            _store_match_block(
                index_1,
                index_2,
                correlation,
                reversed_match,
                selected,
                i_slice=i_slice,
                j_slice=j_slice,
                reversed_selected=block_reversed,
                transpose=False,
            )

            # For non-diagonal blocks, also store results for the lower triangular
            # part of the matrix
            if i_start != j_start:
                _store_match_block(
                    index_1,
                    index_2,
                    correlation,
                    reversed_match,
                    selected,
                    i_slice=j_slice,
                    j_slice=i_slice,
                    reversed_selected=block_reversed,
                    transpose=True,
                )

    return CommonLineGraph(
        index_1=index_1,
        index_2=index_2,
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
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_left: int,
    n_right: int,
    n_angles: int,
) -> _MatchResult:
    """Find the best angular match for every image pair in two blocks."""
    correlation = left @ right.T
    index_1, index_2, score = _reduce_correlation_block(
        correlation,
        n_left=n_left,
        n_right=n_right,
        n_angles=n_angles,
    )
    return _MatchResult(index_1, index_2, score)


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


def _reduce_correlation_block(
    correlation: np.ndarray,
    *,
    n_left: int,
    n_right: int,
    n_angles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the best angular pair for every image pair in one block.

    The matrix is viewed as
    ``(left_image, left_angle, right_image, right_angle)`` and reduced in two
    stages. This avoids transposing and copying all ``(A, A)`` pair matrices.
    """
    correlation = correlation.reshape(
        n_left,
        n_angles,
        n_right,
        n_angles,
    )

    best_right_for_left = np.argmax(correlation, axis=3)
    best_score_for_left = np.take_along_axis(
        correlation,
        best_right_for_left[..., None],
        axis=3,
    )[..., 0]

    index_left = np.argmax(best_score_for_left, axis=1)
    gather_left = index_left[:, None, :]

    index_right = np.take_along_axis(
        best_right_for_left,
        gather_left,
        axis=1,
    )[:, 0, :]
    score = np.take_along_axis(
        best_score_for_left,
        gather_left,
        axis=1,
    )[:, 0, :]

    return index_left, index_right, score


def _choose_best_matches(
    normal: _MatchResult,
    reversed_match: _MatchResult,
    *,
    n_angles_2: int,
) -> tuple[_MatchResult, np.ndarray]:
    """Select between normal and detector-reversed candidates.

    Ties reproduce an elementwise maximum followed by a C-order flattened
    ``argmax``: the earliest angular pair wins, and the normal candidate is
    preferred when both candidates refer to the same pair.
    """
    normal_flat = normal.index_1 * n_angles_2 + normal.index_2
    reversed_flat = reversed_match.index_1 * n_angles_2 + reversed_match.index_2

    use_reversed = reversed_match.correlation > normal.correlation
    use_reversed |= (reversed_match.correlation == normal.correlation) & (
        reversed_flat < normal_flat
    )

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


def _store_match_block(
    index_1: np.ndarray,
    index_2: np.ndarray,
    correlation: np.ndarray,
    reversed_match: np.ndarray | None,
    selected: _MatchResult,
    *,
    i_slice: slice,
    j_slice: slice,
    reversed_selected: np.ndarray | None,
    transpose: bool,
) -> None:
    """Store one block of common-line matches in the global result arrays.

    The input ``selected`` contains the best angular match for every pair of
    images in two blocks. Its arrays have shape ``(n_left, n_right)`` and are
    written into the region defined by ``i_slice`` and ``j_slice``.

    When ``transpose`` is false, the matches are stored directly. When it is
    true, the opposite directed image pairs are stored: the image-pair axes are
    transposed and the angular indices are exchanged, so a match ``(a, b)``
    for ``(i, j)`` becomes ``(b, a)`` for ``(j, i)``.

    Parameters
    ----------
    index_1, index_2
        Global arrays with shape ``(n_images, n_images)`` storing the selected
        indices in the first and second image.
    correlation
        Global array with shape ``(n_images, n_images)`` storing selected
        correlation values.
    reversed_match
        Global boolean array, or ``None`` when reversed profiles are disabled.
    selected
        Best indices and correlations for the current block.
    i_slice, j_slice
        Destination region in the global arrays.
    reversed_selected
        Boolean flags for the current block. It is provided when reversed
        profiles are enabled.
    transpose
        Store the direct block when false, or the opposite-direction block
        when true.

    Notes
    -----
    The output arrays are modified in place. Reusing the transposed block
    avoids recomputing correlations for both directed image-pair orders.
    """
    if transpose:
        index_1[i_slice, j_slice] = selected.index_2.T
        index_2[i_slice, j_slice] = selected.index_1.T
        correlation[i_slice, j_slice] = selected.correlation.T

        if reversed_match is not None:
            if reversed_selected is None:
                raise ValueError(
                    "reversed_match and reverse_selected must be provided together"
                )
            reversed_match[i_slice, j_slice] = reversed_selected.T
        return

    index_1[i_slice, j_slice] = selected.index_1
    index_2[i_slice, j_slice] = selected.index_2
    correlation[i_slice, j_slice] = selected.correlation

    if reversed_match is not None:
        reversed_match[i_slice, j_slice] = reversed_selected


def _indices_to_angles(
    index_1: np.ndarray,
    index_2: np.ndarray,
    *,
    theta_1: np.ndarray | None,
    theta_2: np.ndarray | None,
    reversed_match: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Convert selected indices into oriented angular values."""
    angle_1 = None if theta_1 is None else np.asarray(theta_1)[index_1]
    angle_2 = None if theta_2 is None else np.asarray(theta_2)[index_2]

    if angle_2 is not None and reversed_match is not None:
        angle_2 = np.mod(angle_2 + 180.0 * reversed_match, 360.0)

    return angle_1, angle_2


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


def _validate_theta(
    theta: np.ndarray | None,
    n_angles: int,
    name: str,
) -> None:
    """Check that an optional angle grid has the expected one-dimensional shape."""
    if theta is None:
        return

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
