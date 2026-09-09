from __future__ import annotations

import numpy as np

from common_lines.centering._validation import _as_image_batch, _as_single_image
from common_lines.centering.shifts import shift_image
from .preprocessing import preprocess_images, PreprocessingOptions


def estimate_reflection_shift(
    image: np.ndarray, *, preprocessing_options: PreprocessingOptions | None = None
) -> tuple[float, float]:
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
    preprocessing_options : PreprocessingOptions
        Parameters for ``preprocess_images``. See ``PreprocessingOptions``
        for more information.

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
    shift = estimate_reflection_shifts(
        image[None, ...], preprocessing_options=preprocessing_options
    )[0]
    return float(shift[0]), float(shift[1])


def estimate_reflection_shifts(
    images: np.ndarray, *, preprocessing_options: PreprocessingOptions | None = None
) -> np.ndarray:
    """
    Estimate centering corrections for a batch of images.

    Both reflections are formed from the original images. Each shift component
    is estimated independently along its own axis, and both estimates are made
    before any interpolation is applied.

    Parameters
    ----------
    images
        Real-valued array with shape ``(batch, height, width)``.
    preprocessing_options : PreprocessingOptions
        Parameters for ``preprocess_images``. See ``PreprocessingOptions``
        for more information.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(batch, 2)``. Column 0 contains ``shift_y`` and
        column 1 contains ``shift_x``. The values are the translations that
        should be applied to the input images.
    """
    images = _as_image_batch(images)
    batch_size = images.shape[0]

    processed = preprocess_images(images, options=preprocessing_options)

    # Removing the DC component prevents a constant background from dominating
    # the correlation with the reflected images.
    centered = processed - processed.mean(axis=(-2, -1), keepdims=True)

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


def precenter_image_reflection(
    image: np.ndarray, preprocessing_options: PreprocessingOptions | None = None
) -> np.ndarray:
    """Approximately center one image using reflection correlations."""
    image = _as_single_image(image)
    shift_y, shift_x = estimate_reflection_shift(
        image, preprocessing_options=preprocessing_options
    )
    return shift_image(image, shift_y, shift_x)


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
