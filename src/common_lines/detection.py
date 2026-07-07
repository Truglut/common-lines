from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommonLineResult:
    """
    Result of common-line detection between sinogram pairs.

    Attributes
    ----------
    index_1
        Best column index in the first sinogram.
    index_2
        Best column index in the second sinogram.
    correlation
        Pearson correlation value for the selected common line.
    angle_1
        Angle values corresponding to index_1, if theta_1 was provided.
    angle_2
        Angle values corresponding to index_2, if theta_2 was provided.
    reversed
        Whether the best match used a reversed detector direction.
        Only meaningful when allow_reversed=True.
    correlation_matrix
        Optional full correlation matrix/matrices.
        For pairwise batches, shape is (B, A1, A2).
    """

    index_1: np.ndarray
    index_2: np.ndarray
    correlation: np.ndarray
    angle_1: np.ndarray | None = None
    angle_2: np.ndarray | None = None
    reversed: np.ndarray | None = None
    correlation_matrix: np.ndarray | None = None


def detect_common_lines(
    sinograms_1: np.ndarray,
    sinograms_2: np.ndarray,
    *,
    theta_1: np.ndarray | None = None,
    theta_2: np.ndarray | None = None,
    eps: float = 1.0e-8,
    allow_reversed: bool = False,
    return_correlation_matrix: bool = False,
) -> CommonLineResult:
    """
    Detect common lines between pairs of sinograms.

    Parameters
    ----------
    sinograms_1
        First sinogram or batch of sinograms, with shape (P, A1) or (B, P, A1).
    sinograms_2
        Second sinogram or batch of sinograms, with shape (P, A2) or (B, P, A2).
    theta_1
        Optional angle array for sinograms_1, with shape (A1,).
    theta_2
        Optional angle array for sinograms_2, with shape (A2,).
    eps
        Small numerical tolerance used when normalizing columns.
    allow_reversed
        If True, also compares columns after reversing the detector axis of the
        second sinogram. This is useful when the same line may appear with
        opposite orientation.
    return_correlation_matrix
        If True, include the full pairwise correlation matrix in the result.

    Returns
    -------
    CommonLineResult
        Contains the best column pair and its correlation for each sinogram pair.

    Notes
    -----
    This function compares batch element b in sinograms_1 with batch element b
    in sinograms_2. If one input contains a single sinogram and the other is a
    batch, the single sinogram is broadcast against the full batch.
    """
    # Convert inputs to batch form and broadcast if necessary
    s1 = _as_batch_sinograms(sinograms_1)
    s2 = _as_batch_sinograms(sinograms_2)

    s1, s2 = _broadcast_batches(s1, s2)
    
    batch_size, _, n_angles_1 = s1.shape
    _, _, n_angles_2 = s2.shape

    # Input validation
    if s1.shape[1] != s2.shape[1]:
        raise ValueError(
            "sinograms_1 and sinograms_2 must have the same detector dimension. "
            f"Got {s1.shape[1]} and {s2.shape[1]}."
        )

    _validate_theta(theta_1, n_angles_1, "theta_1")
    _validate_theta(theta_2, n_angles_2, "theta_2")

    # Normalize columns and compute correlation matrix
    z1 = _normalize_columns(s1, eps=eps)
    z2 = _normalize_columns(s2, eps=eps)

    # corr[b, a, c] = sum_p z1[b, p, a] * z2[b, p, c]
    corr = np.einsum("bpa,bpc->bac", z1, z2, optimize=True)
    score = corr
    reversed_best_matrix = None

    # If allow_reversed is True, also compute correlations with reversed sinograms
    if allow_reversed:
        z2_reversed = z2[:, ::-1, :]
        corr_reversed = np.einsum("bpa,bpc->bac", z1, z2_reversed, optimize=True)

        reversed_best_matrix = corr_reversed > corr
        score = np.where(reversed_best_matrix, corr_reversed, corr)

    # Find the best matching columns for each batch element
    flat_index = np.argmax(score.reshape(batch_size, -1), axis=1)

    # Map flat indices back to 2D indices
    index_1 = flat_index // n_angles_2
    index_2 = flat_index % n_angles_2

    # Get best correlation values for each batch element
    batch_indices = np.arange(batch_size)
    best_correlation = score[batch_indices, index_1, index_2]

    # Map indices to angles if theta arrays were provided
    angle_1 = None if theta_1 is None else np.asarray(theta_1)[index_1]
    angle_2 = None if theta_2 is None else np.asarray(theta_2)[index_2]

    # If allow_reversed is True, get the reversed selection for each batch element
    reversed_selected = None
    if allow_reversed:
        reversed_selected = reversed_best_matrix[batch_indices, index_1, index_2]

    # If return_correlation_matrix is True, include the full correlation matrix in the result
    correlation_matrix = score if return_correlation_matrix else None

    return CommonLineResult(
        index_1=index_1,
        index_2=index_2,
        correlation=best_correlation,
        angle_1=angle_1,
        angle_2=angle_2,
        reversed=reversed_selected,
        correlation_matrix=correlation_matrix,
    )


def detect_common_lines_all_pairs(
    sinograms: np.ndarray,
    *,
    theta: np.ndarray | None = None,
    eps: float = 1.0e-8,
    allow_reversed: bool = False,
    return_correlation_tensor: bool = False,
) -> CommonLineResult:
    """
    Detect common lines for all pairs inside a batch of sinograms.

    Parameters
    ----------
    sinograms
        Batch of sinograms with shape (B, P, A).
    theta
        Optional angle array with shape (A,).
    eps
        Small numerical tolerance used when normalizing columns.
    allow_reversed
        If True, also compares columns after reversing the detector axis of the
        second sinogram.
    return_correlation_tensor
        If True, include the full correlation tensor with shape (B, B, A, A).

    Returns
    -------
    CommonLineResult
        Result matrices with shape (B, B).

    Notes
    -----
    This function is fully vectorized but can be memory-heavy because the full
    correlation tensor has shape (B, B, A, A). For large batches, prefer
    detect_common_lines on explicit batches of image pairs.
    """
    # Input preparation and validation

    s = np.asarray(sinograms)
    if s.ndim != 3:
        raise ValueError(
            "sinograms must have shape (B, P, A). "
            f"Got array with shape {s.shape}."
        )
    
    batch_size, _, n_angles = s.shape

    _validate_theta(theta, n_angles, "theta")

    # Compute correlation tensor
    z = _normalize_columns(s, eps=eps)
    corr = np.einsum("ipa,jpb->ijab", z, z, optimize=True)
    
    # If allow_reversed is True, also compute correlations with reversed sinograms
    score = corr
    reversed_best_tensor = None
    if allow_reversed:
        z_reversed = z[:, ::-1, :]
        corr_reversed = np.einsum("ipa,jpb->ijab", z, z_reversed, optimize=True)

        reversed_best_tensor = corr_reversed > corr
        score = np.where(reversed_best_tensor, corr_reversed, corr)

    # Get best matching columns and map flat indices back to 2D indices
    flat_index = np.argmax(score.reshape(batch_size, batch_size, -1), axis=-1)
    index_1 = flat_index // n_angles
    index_2 = flat_index % n_angles

    row_indices = np.arange(batch_size)[:, None]
    col_indices = np.arange(batch_size)[None, :]

    # Prepare output values
    best_correlation = score[row_indices, col_indices, index_1, index_2]

    angle_1 = None if theta is None else np.asarray(theta)[index_1]
    angle_2 = None if theta is None else np.asarray(theta)[index_2]

    reversed_selected = None
    if allow_reversed:
        reversed_selected = reversed_best_tensor[
            row_indices,
            col_indices,
            index_1,
            index_2,
        ]

    correlation_matrix = score if return_correlation_tensor else None

    return CommonLineResult(
        index_1=index_1,
        index_2=index_2,
        correlation=best_correlation,
        angle_1=angle_1,
        angle_2=angle_2,
        reversed=reversed_selected,
        correlation_matrix=correlation_matrix,
    )


def _as_batch_sinograms(sinograms: np.ndarray) -> np.ndarray:
    """
    Convert a single sinogram to a batch of one sinogram.

    Expected formats:
      - (P, A) for a single sinogram
      - (B, P, A) for a batch
    """
    sinograms = np.asarray(sinograms)

    if sinograms.ndim == 2:
        sinograms = sinograms[None, ...]
    elif sinograms.ndim != 3:
        raise ValueError(
            "sinograms must have shape (P, A) or (B, P, A). "
            f"Got array with shape {sinograms.shape}."
        )

    # Integer arrays are converted because the normalization step needs floating
    # point arithmetic.
    if not np.issubdtype(sinograms.dtype, np.floating):
        sinograms = sinograms.astype(np.float64, copy=False)

    return sinograms


def _broadcast_batches(
    s1: np.ndarray,
    s2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make batch sizes compatible.

    Supported cases:
      - both inputs have the same batch size
      - s1 has batch size 1 and s2 has batch size B
      - s2 has batch size 1 and s1 has batch size B
    """
    b1 = s1.shape[0]
    b2 = s2.shape[0]

    if b1 == b2:
        return s1, s2

    if b1 == 1:
        return np.broadcast_to(s1, (b2, *s1.shape[1:])), s2

    if b2 == 1:
        return s1, np.broadcast_to(s2, (b1, *s2.shape[1:]))

    raise ValueError(
        "Batch dimensions must match, unless one input contains a single "
        f"sinogram. Got batch sizes {b1} and {b2}."
    )


def _normalize_columns(sinograms: np.ndarray, *, eps: float) -> np.ndarray:
    """
    Center and normalize each sinogram column.

    For each column x, this computes:

        z = (x - mean(x)) / ||x - mean(x)||

    After this transformation, dot(z1, z2) is the Pearson correlation between
    the original columns.

    Constant columns have zero norm after centering. Those are left as zero
    columns, giving correlation zero against every other column.
    """
    # Subtract the mean along the detector/profile axis.
    centered = sinograms - sinograms.mean(axis=1, keepdims=True)

    # Compute one norm per column.
    norms = np.linalg.norm(centered, axis=1, keepdims=True)

    # Fill the output safely. Columns whose norm is too small remain zero.
    normalized = np.zeros_like(centered)
    np.divide(centered, norms, out=normalized, where=norms > eps)

    return normalized


def _validate_theta(
    theta: np.ndarray | None,
    n_angles: int,
    name: str,
) -> None:
    """
    Check that an angle array has the expected length.
    """
    if theta is None:
        return

    theta = np.asarray(theta)

    if theta.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if theta.shape[0] != n_angles:
        raise ValueError(
            f"{name} must have length {n_angles}. "
            f"Got length {theta.shape[0]}."
        )