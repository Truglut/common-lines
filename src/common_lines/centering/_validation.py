import numpy as np


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
