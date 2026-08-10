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
    The estimator uses circular FFT correlation. It is intended as a
    pre-centering step, so the particle should already lie reasonably close to
    the image center. Because reflection doubles the displacement, shifts
    approaching one quarter of the image size along an axis become ambiguous.
    """
    image = _as_single_image(image)
    shift = estimate_center_shifts(image[None, ...])[0]
    return float(shift[0]), float(shift[1])


def estimate_center_shifts(images: np.ndarray) -> np.ndarray:
    """
    Estimate centering corrections for a batch of images.

    Both reflections are formed from the original images. FFTs and correlation
    maps are computed for the complete batch at once.

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
    image_fft = np.fft.rfftn(centered, axes=(-2, -1))

    lag_x = _reflection_lag(
        centered,
        image_fft,
        reflection_axis=-1,
        lag_axis=-1,
    )
    lag_y = _reflection_lag(
        centered,
        image_fft,
        reflection_axis=-2,
        lag_axis=-2,
    )

    shifts = np.empty((batch_size, 2), dtype=np.float64)
    shifts[:, 0] = -0.5 * lag_y
    shifts[:, 1] = -0.5 * lag_x

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
    image_fft: np.ndarray,
    *,
    reflection_axis: int,
    lag_axis: int,
) -> np.ndarray:
    """Return the subpixel correlation lag against one image reflection."""
    reflected_fft = np.fft.rfftn(
        np.flip(centered, axis=reflection_axis),
        axes=(-2, -1),
    )
    correlation = np.fft.irfftn(
        image_fft * np.conj(reflected_fft),
        s=centered.shape[-2:],
        axes=(-2, -1),
    )

    batch_size, height, width = correlation.shape
    flat_peak = np.argmax(correlation.reshape(batch_size, -1), axis=1)
    peak_y, peak_x = np.divmod(flat_peak, width)

    if lag_axis == -1:
        integer_lag = _wrapped_indices_to_lags(peak_x, width)
    elif lag_axis == -2:
        integer_lag = _wrapped_indices_to_lags(peak_y, height)
    else:
        raise ValueError("lag_axis must be -1 or -2.")

    fractional_lag = _parabolic_peak_offset(
        correlation,
        peak_y,
        peak_x,
        axis=lag_axis,
    )
    return integer_lag.astype(np.float64) + fractional_lag


def _parabolic_peak_offset(
    correlation: np.ndarray,
    peak_y: np.ndarray,
    peak_x: np.ndarray,
    *,
    axis: int,
) -> np.ndarray:
    """Refine a discrete correlation maximum with a three-point parabola."""
    batch = np.arange(correlation.shape[0])
    center = correlation[batch, peak_y, peak_x]

    if axis == -1:
        minus = correlation[batch, peak_y, (peak_x - 1) % correlation.shape[2]]
        plus = correlation[batch, peak_y, (peak_x + 1) % correlation.shape[2]]
    elif axis == -2:
        minus = correlation[batch, (peak_y - 1) % correlation.shape[1], peak_x]
        plus = correlation[batch, (peak_y + 1) % correlation.shape[1], peak_x]
    else:
        raise ValueError("axis must be -1 or -2.")

    denominator = minus - 2.0 * center + plus
    offset = np.zeros(correlation.shape[0], dtype=np.float64)

    scale = np.maximum(1.0, np.abs(center))
    valid = np.abs(denominator) > np.finfo(correlation.dtype).eps * scale
    offset[valid] = 0.5 * (minus[valid] - plus[valid]) / denominator[valid]

    # For a genuine local maximum the parabolic vertex lies in this interval.
    # Clipping also makes the refinement harmless for noisy, poorly shaped peaks.
    return np.clip(offset, -0.5, 0.5)


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
