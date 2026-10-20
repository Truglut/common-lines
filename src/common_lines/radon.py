from __future__ import annotations

import warnings

import numpy as np
from numba import njit, prange


def _batch_radon(
    images: np.ndarray,
    theta: np.ndarray | None = None,
    *,
    circle: bool = True,
    warn_if_nonzero_outside_circle: bool = False,
) -> np.ndarray:
    """
    Compute Radon sinograms for a full batch of 2D images.

    This is designed to match the convention used by:

        skimage.transform.radon(image, theta=theta, circle=circle,
                                preserve_range=True)

    Parameters
    ----------
    images
        Array of shape (B, H, W), or a single image of shape (H, W).
    theta
        Projection angles in degrees. If None, np.arange(180) is used.
    circle
        Same meaning as in skimage.transform.radon.

        If True, the projection width is min(H, W). Rectangular images are
        center-cropped to a square of side min(H, W).

        If False, images are padded to fit the full rotated image.
    warn_if_nonzero_outside_circle
        If True and circle=True, issue a warning when pixels outside the
        reconstruction circle are nonzero, following the spirit of skimage.

    Returns
    -------
    sinograms
        If images has shape (B, H, W), returns shape (B, P, A), where
        P is the projection width and A = len(theta).

        If images has shape (H, W), returns shape (P, A).
    """
    input_ndim = np.asarray(images).ndim

    images = np.asarray(images)
    if images.ndim == 2:
        images = images[None, ...]
    elif images.ndim != 3:
        raise ValueError("images must have shape (H, W) or (B, H, W).")

    if theta is None:
        theta = np.arange(180, dtype=np.float64)
    else:
        theta = np.asarray(theta, dtype=np.float64)

    if not np.issubdtype(images.dtype, np.floating):
        images = images.astype(np.float64, copy=False)

    images = np.ascontiguousarray(images)

    if circle and warn_if_nonzero_outside_circle:
        _warn_if_nonzero_outside_circle(images)

    prepared = _prepare_images_like_skimage(images, circle=circle)
    prepared = np.ascontiguousarray(prepared)

    theta_rad = np.deg2rad(theta)
    sinograms = _batch_radon_square_numba(prepared, theta_rad)

    if input_ndim == 2:
        return sinograms[0]
    return sinograms


def _prepare_images_like_skimage(images: np.ndarray, *, circle: bool) -> np.ndarray:
    """
    Apply the same crop/pad convention used by skimage.transform.radon.

    The output is always a batch of square images.
    """
    _, height, width = images.shape

    if circle:
        shape_min = min(height, width)
        img_shape = np.array([height, width])

        slices = tuple(
            (
                slice(
                    int(np.ceil(excess / 2)),
                    int(np.ceil(excess / 2) + shape_min),
                )
                if excess > 0
                else slice(None)
            )
            for excess in (img_shape - shape_min)
        )

        return images[:, slices[0], slices[1]]

    diagonal = np.sqrt(2.0) * max(height, width)
    pad = [int(np.ceil(diagonal - size)) for size in (height, width)]

    new_center = [(size + p) // 2 for size, p in zip((height, width), pad)]
    old_center = [size // 2 for size in (height, width)]

    pad_before = [new_c - old_c for old_c, new_c in zip(old_center, new_center)]

    pad_width = [(before, total - before) for before, total in zip(pad_before, pad)]

    return np.pad(
        images,
        ((0, 0), pad_width[0], pad_width[1]),
        mode="constant",
        constant_values=0,
    )


def _warn_if_nonzero_outside_circle(images: np.ndarray) -> None:
    _, height, width = images.shape
    shape_min = min(height, width)
    radius = shape_min // 2

    yy, xx = np.ogrid[:height, :width]
    center = np.array([height // 2, width // 2])

    outside = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 > radius**2

    if np.any(images[:, outside]):
        warnings.warn(
            "Radon transform: image must be zero outside the reconstruction circle",
            UserWarning,
            stacklevel=2,
        )


@njit(parallel=True, cache=True)
def _batch_radon_square_numba(
    images: np.ndarray,
    theta_rad: np.ndarray,
) -> np.ndarray:
    """
    Numba kernel for a batch of square images.

    images shape: (B, S, S)
    output shape: (B, S, A)

    The implementation follows the same affine rotation convention as
    skimage.transform.radon: rotate image, then sum over rows.
    """
    batch_size, size, _ = images.shape
    n_angles = theta_rad.shape[0]

    out = np.empty((batch_size, size, n_angles), dtype=images.dtype)
    center = size // 2

    n_jobs = batch_size * size * n_angles

    for idx in prange(n_jobs):
        angle_idx = idx % n_angles
        detector_idx = (idx // n_angles) % size
        batch_idx = idx // (n_angles * size)

        angle = theta_rad[angle_idx]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        col_base = cos_a * detector_idx - center * (cos_a + sin_a - 1.0)
        row_base = -sin_a * detector_idx - center * (cos_a - sin_a - 1.0)

        total = 0.0

        for row_out in range(size):
            src_col = col_base + sin_a * row_out
            src_row = row_base + cos_a * row_out

            row0 = int(np.floor(src_row))
            col0 = int(np.floor(src_col))

            wr = src_row - row0
            wc = src_col - col0

            row1 = row0 + 1
            col1 = col0 + 1

            if 0 <= row0 < size:
                if 0 <= col0 < size:
                    total += images[batch_idx, row0, col0] * (1.0 - wr) * (1.0 - wc)
                if 0 <= col1 < size:
                    total += images[batch_idx, row0, col1] * (1.0 - wr) * wc

            if 0 <= row1 < size:
                if 0 <= col0 < size:
                    total += images[batch_idx, row1, col0] * wr * (1.0 - wc)
                if 0 <= col1 < size:
                    total += images[batch_idx, row1, col1] * wr * wc

        out[batch_idx, detector_idx, angle_idx] = total

    return out


def batch_common_line_sinograms(
    images: np.ndarray,
    theta: np.ndarray | None = None,
    *,
    circle: bool = True,
    warn_if_nonzero_outside_circle: bool = False,
) -> np.ndarray:
    """
    Compute sinograms using the angular convention expected by the common-line
    orientation code.

    In this convention, image coordinates are ``(x, y) = (column, row)`` and
    positive y points towards increasing row indices. A sinogram column labelled
    by theta therefore corresponds to the oriented image-plane direction

        (cos(theta), sin(theta)).

    The lower-level batch_radon function follows the skimage-compatible rotation
    convention, so the sign is converted here in one place. Angles in [0, 180)
    describe the unoriented lines; profile reversal can later be represented by
    adding 180 degrees to the corresponding angle.
    """
    if theta is None:
        theta = np.arange(180.0)
    else:
        theta = np.asarray(theta, dtype=np.float64)

    return _batch_radon(
        images,
        theta=-theta,
        circle=circle,
        warn_if_nonzero_outside_circle=warn_if_nonzero_outside_circle,
    )
