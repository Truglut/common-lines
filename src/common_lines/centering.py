from __future__ import annotations

import numpy as np


def estimate_center_shift(image: np.ndarray) -> tuple[float, float]:
    """
    Estimate the translational correction that approximately centers one image.

    The horizontal correction is estimated by correlating the image with its
    left-right reflection. The vertical correction is estimated independently
    from the original image and its up-down reflection. A displacement of the
    particle by ``d`` produces a relative displacement of approximately ``2*d``
    with respect to the corresponding reflection, so the measured lag is
    divided by two and its sign is reversed.

    Parameters
    ----------
    image
        Real-valued image with shape ``(height, width)``.

    Returns
    -------
    shift_y, shift_x
        Translation, in pixels, that should be applied to the image to move it
        towards the center. Fractional values are allowed.

    Notes
    -----
    The estimator uses circular FFT correlation along each image axis. The
    target center is the pixel with index ``size // 2`` along each dimension.
    For square images this matches the rotation center used by the common-line
    Radon transform. It is intended as a pre-centering step, so the particle
    should already lie reasonably close to the image center. 
    Because reflection doubles the displacement, shifts approaching one quarter 
    of the image size along an axis become ambiguous.
    """
    image = _as_single_image(image)
    shift = estimate_center_shifts(image[None, ...])[0]
    return float(shift[0]), float(shift[1])


def estimate_center_shifts(images: np.ndarray) -> np.ndarray:
    """
    Estimate centering corrections for a batch of images.

    Both reflections are formed from the original images. Each shift component
    is estimated independently along its own axis, and both estimates are made
    before any interpolation is applied.

    Parameters
    ----------
    images
        Real-valued array with shape ``(batch, height, width)``.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(batch, 2)``. Column 0 contains ``shift_y`` and
        column 1 contains ``shift_x``. The values are the translations that
        should be applied to the input images.
    """
    images = _as_image_batch(images)
    batch_size = images.shape[0]

    # Removing the DC component prevents a constant background from dominating
    # the correlation with the reflected images.
    centered = images - images.mean(axis=(-2, -1), keepdims=True)

    # Estimate each component independently from the original images. For the
    # left-right reflection, only horizontal lags are allowed; for the up-down
    # reflection, only vertical lags are allowed.
    lag_x = _reflection_lag(centered, axis=-1)
    lag_y = _reflection_lag(centered, axis=-2)

    shifts = np.empty((batch_size, 2), dtype=np.float64)
    shifts[:, 0] = -0.5 * lag_y + _pixel_center_offset(images.shape[-2])
    shifts[:, 1] = -0.5 * lag_x + _pixel_center_offset(images.shape[-1])

    # A constant image contains no centering information. Keep it unchanged
    # rather than applying the half-pixel convention correction for even sizes.
    informative = np.any(centered != 0.0, axis=(-2, -1))
    shifts[~informative] = 0.0

    return shifts


def shift_image(
    image: np.ndarray,
    shift_y: float,
    shift_x: float,
) -> np.ndarray:
    """
    Translate one image without wrapping values across its borders.

    Bilinear interpolation is used for fractional shifts and pixels sampled
    outside the input image are set to zero.
    """
    image = _as_single_image(image)
    shifts = np.array([[shift_y, shift_x]], dtype=np.float64)
    return shift_images(image[None, ...], shifts)[0]


def shift_images(images: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """
    Translate a batch of images using vectorized bilinear interpolation.

    Parameters
    ----------
    images
        Array with shape ``(batch, height, width)``.
    shifts
        Array with shape ``(batch, 2)`` containing ``(shift_y, shift_x)`` for
        each image. Positive values move image content towards increasing row
        or column indices.

    Returns
    -------
    numpy.ndarray
        Shifted images with the same shape as ``images``. Floating-point inputs
        retain their precision; integer inputs are converted to ``float64``.
    """
    images = _as_image_batch(images)
    shifts = np.asarray(shifts, dtype=np.float64)

    if shifts.shape != (images.shape[0], 2):
        raise ValueError(
            "shifts must have shape (batch, 2). "
            f"Expected {(images.shape[0], 2)}, got {shifts.shape}."
        )
    if not np.all(np.isfinite(shifts)):
        raise ValueError("shifts must contain only finite values.")

    batch_size, height, width = images.shape

    # Horizontal interpolation. Coordinates are source coordinates for each
    # output column, hence the subtraction of the requested image shift.
    source_x = np.arange(width, dtype=np.float64)[None, :] - shifts[:, 1, None]
    x0 = np.floor(source_x).astype(np.int64)
    x1 = x0 + 1
    wx = (source_x - x0).astype(images.dtype, copy=False)

    valid_x0 = (x0 >= 0) & (x0 < width)
    valid_x1 = (x1 >= 0) & (x1 < width)
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)

    index_x0 = np.broadcast_to(x0[:, None, :], (batch_size, height, width))
    index_x1 = np.broadcast_to(x1[:, None, :], (batch_size, height, width))
    value_x0 = np.take_along_axis(images, index_x0, axis=2)
    value_x1 = np.take_along_axis(images, index_x1, axis=2)

    horizontal = (1.0 - wx[:, None, :]) * value_x0 * valid_x0[:, None, :] + wx[
        :, None, :
    ] * value_x1 * valid_x1[:, None, :]

    # Vertical interpolation is applied to the horizontally shifted batch.
    source_y = np.arange(height, dtype=np.float64)[None, :] - shifts[:, 0, None]
    y0 = np.floor(source_y).astype(np.int64)
    y1 = y0 + 1
    wy = (source_y - y0).astype(images.dtype, copy=False)

    valid_y0 = (y0 >= 0) & (y0 < height)
    valid_y1 = (y1 >= 0) & (y1 < height)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    index_y0 = np.broadcast_to(y0[:, :, None], (batch_size, height, width))
    index_y1 = np.broadcast_to(y1[:, :, None], (batch_size, height, width))
    value_y0 = np.take_along_axis(horizontal, index_y0, axis=1)
    value_y1 = np.take_along_axis(horizontal, index_y1, axis=1)

    shifted = (1.0 - wy[:, :, None]) * value_y0 * valid_y0[:, :, None] + wy[
        :, :, None
    ] * value_y1 * valid_y1[:, :, None]

    return shifted.astype(images.dtype, copy=False)


def precenter_image(image: np.ndarray) -> np.ndarray:
    """Approximately center one image using reflection correlations."""
    image = _as_single_image(image)
    shift_y, shift_x = estimate_center_shift(image)
    return shift_image(image, shift_y, shift_x)


def precenter_images(images: np.ndarray) -> np.ndarray:
    """
    Approximately center a full batch of images.

    Shift estimation and interpolation are both performed in batch. The
    horizontal and vertical shifts are estimated from the same original input
    images, and a single 2D translation is then applied to each image.
    """
    images = _as_image_batch(images)
    shifts = estimate_center_shifts(images)
    return shift_images(images, shifts)


def _reflection_lag(
    centered: np.ndarray,
    *,
    axis: int,
) -> np.ndarray:
    """Return the subpixel lag against a reflection along one image axis."""
    if axis not in (-1, -2):
        raise ValueError("axis must be -1 or -2.")

    size = centered.shape[axis]
    reflected = np.flip(centered, axis=axis)

    # Correlate only along the reflected axis. Summing over the orthogonal
    # dimension is equivalent to taking the zero-lag slice of the full 2D
    # correlation, so an auxiliary shift in the other direction cannot change
    # which lag is selected.
    image_fft = np.fft.rfft(centered, axis=axis)
    reflected_fft = np.fft.rfft(reflected, axis=axis)
    correlation = np.fft.irfft(
        image_fft * np.conj(reflected_fft),
        n=size,
        axis=axis,
    )

    if axis == -1:
        correlation = correlation.sum(axis=-2)
    else:
        correlation = correlation.sum(axis=-1)

    peak = np.argmax(correlation, axis=1)
    integer_lag = _wrapped_indices_to_lags(peak, size)
    fractional_lag = _parabolic_peak_offset(correlation, peak)

    return integer_lag.astype(np.float64) + fractional_lag


def _parabolic_peak_offset(
    correlation: np.ndarray,
    peak: np.ndarray,
) -> np.ndarray:
    """Refine a one-dimensional correlation maximum with a parabola."""
    batch = np.arange(correlation.shape[0])
    size = correlation.shape[1]

    center = correlation[batch, peak]
    minus = correlation[batch, (peak - 1) % size]
    plus = correlation[batch, (peak + 1) % size]

    denominator = minus - 2.0 * center + plus
    offset = np.zeros(correlation.shape[0], dtype=np.float64)

    scale = np.maximum(1.0, np.abs(center))
    valid = np.abs(denominator) > np.finfo(correlation.dtype).eps * scale
    offset[valid] = 0.5 * (minus[valid] - plus[valid]) / denominator[valid]

    # For a genuine local maximum the parabolic vertex lies in this interval.
    # Clipping also makes the refinement harmless for noisy, poorly shaped peaks.
    return np.clip(offset, -0.5, 0.5)


def _pixel_center_offset(size: int) -> float:
    """Return the correction from the flip center to pixel index ``size // 2``."""
    return size // 2 - (size - 1) / 2.0


def _wrapped_indices_to_lags(indices: np.ndarray, size: int) -> np.ndarray:
    """Convert circular-correlation indices to signed integer lags."""
    half = size // 2
    return ((indices + half) % size) - half


def _as_single_image(image: np.ndarray) -> np.ndarray:
    """Validate and convert one image."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            "image must have shape (height, width). "
            f"Got array with shape {image.shape}."
        )
    return _validate_and_convert_images(image[None, ...])[0]


def _as_image_batch(images: np.ndarray) -> np.ndarray:
    """Validate and convert an explicit image batch."""
    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError(
            "images must have shape (batch, height, width). "
            f"Got array with shape {images.shape}."
        )
    return _validate_and_convert_images(images)


def _validate_and_convert_images(images: np.ndarray) -> np.ndarray:
    """Reject invalid data and prepare a floating-point image array."""
    if any(size == 0 for size in images.shape):
        raise ValueError("images must not contain empty dimensions.")
    if images.shape[-2] < 2 or images.shape[-1] < 2:
        raise ValueError("image height and width must both be at least 2.")
    if np.iscomplexobj(images):
        raise TypeError("images must contain real-valued data.")
    if not np.all(np.isfinite(images)):
        raise ValueError("images must contain only finite values.")

    if not np.issubdtype(images.dtype, np.floating):
        return images.astype(np.float64, copy=False)
    if images.dtype.itemsize < np.dtype(np.float32).itemsize:
        return images.astype(np.float32, copy=False)
    return images
