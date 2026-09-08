from common_lines.centering._validation import _as_single_image
from common_lines.centering._validation import _as_image_batch


import numpy as np


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
