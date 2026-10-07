from dataclasses import dataclass

import numpy as np


@dataclass
class PreprocessingOptions:
    """
    Options used for preprocessing particle images.

    Attributes
    ----------
    mask : np.ndarray, optional
        Mask that determines which parts of the images are retained during
        preprocessing.
        It can be an array of shape ``(batch, height, width)`` containing an
        individual mask for each image, or shape ``(height, width)`` to use the
        same mask for every image.
        Its elements should be floating point numbers between 0 and 1.
        The mask is applied after background subtraction and optional negative
        clipping.
        If no mask is provided, then no mask will be applied to the images.
        Default is None.
    subtract_background : bool, optional
        If True, the background level will be estimated for every image using
        the average image value in ``background_mask``, or the average value in
        ``1.0 - mask`` if ``background_mask`` is not provided. This background
        level will then be subtracted from the images before the remaining
        preprocessing steps.
        Default is False.
    background_mask : np.ndarray, optional
        Mask used for the background level estimation. Default is None.
    clip_negative : bool, optional
        If True, any negative values in the images will be clipped to zero.
        Default is False.
    """

    mask: np.ndarray | None = None
    subtract_background: bool = False
    background_mask: np.ndarray | None = None
    clip_negative: bool = False


def default_preprocessing_options(images: np.ndarray) -> PreprocessingOptions:
    """
    Build the standard preprocessing used by the orientation pipeline.

    The default uses an inscribed circular mask, estimates the background from
    the region outside that mask, subtracts that background, and leaves negative
    values unchanged.
    """
    image_shape = np.asarray(images).shape[-2:]
    radius = min(image_shape) / 2.0
    mask = create_circular_mask(image_shape, radius=radius)

    return PreprocessingOptions(
        mask=mask,
        subtract_background=True,
        clip_negative=False,
    )


def preprocess_images(
    images: np.ndarray, options: PreprocessingOptions | None = None
) -> np.ndarray:
    """
    Preprocess an image batch.

    Parameters
    ----------
    images : np.ndarray
        Image batch, array of shape ``(batch, height, width)``.
    options : PreprocessingOptions, optional
        Object containing the different options used in preprocessing.
        See ``PreprocessingOptions`` for more information.
        If not provided, no preprocessing will be done to the images.

    Returns
    -------
    np.ndarray
        Preprocessed images.
    """
    processed = images.astype(dtype=np.float64, copy=True)

    if options is None:
        return processed

    background_mask = _get_background_mask(
        mask=options.mask,
        subtract_background=options.subtract_background,
        background_mask=options.background_mask,
        image_batch_shape=images.shape,
    )

    if options.subtract_background:
        if background_mask is None:
            raise ValueError(
                "Background mask extraction failed: subtract_background was requested "
                "but background_mask is None"
            )

        background = processed * background_mask
        background_mass = background_mask.sum(axis=(-2, -1), keepdims=True)
        background_level = (
            background.sum(axis=(-2, -1), keepdims=True) / background_mass
        )

        processed -= background_level

    if options.clip_negative:
        np.maximum(processed, 0.0, out=processed)

    if options.mask is not None:
        mask = _prepare_mask(options.mask, image_batch_shape=images.shape)
        processed *= mask

    return processed


def _get_background_mask(
    *,
    mask: np.ndarray | None,
    subtract_background: bool,
    background_mask: np.ndarray | None,
    image_batch_shape: tuple[int, int, int],
) -> np.ndarray | None:
    if not subtract_background:
        return None

    if background_mask is None:
        if mask is None:
            raise ValueError("subtract_background was requested but no mask provided.")

        return _prepare_mask(1.0 - mask, image_batch_shape=image_batch_shape)

    return _prepare_mask(background_mask, image_batch_shape=image_batch_shape)


def _prepare_mask(
    mask: np.ndarray, *, image_batch_shape: tuple[int, int, int]
) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float64)

    _, height, width = image_batch_shape

    if not np.all(np.isfinite(mask)):
        raise ValueError(f"Mask must contain only finite values.")

    if np.any(mask < 0.0) or np.any(mask > 1.0):
        raise ValueError(f"Mask values must lie in [0, 1].")

    if mask.ndim == 2:
        if mask.shape != (height, width):
            raise ValueError(
                f"Mask must have shape {(height, width)} or {image_batch_shape}. Got {mask.shape}"
            )
        return np.broadcast_to(mask, image_batch_shape)

    if mask.ndim == 3:
        if mask.shape == image_batch_shape:
            return mask

        if mask.shape == (1, height, width):
            return np.broadcast_to(mask, image_batch_shape)

        raise ValueError(
            f"Mask must have shape {(height, width)} or {image_batch_shape}. Got {mask.shape}"
        )

    raise ValueError(
        f"Mask must be two- or three-dimensional; got array with shape {mask.shape}."
    )


def create_circular_mask(image_shape: tuple[int, int], radius: float) -> np.ndarray:
    """
    Create a circular mask centered in the image.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Shape of the image as (height, width).
    radius : float
        Radius of the circle in pixels.

    Returns
    -------
    np.ndarray
        Boolean array of shape (height, width) where True values fall
        within or on the circle boundary.

    Notes
    -----
        The image center is considered to be at position ``(height // 2, width // 2)``.
    """
    h, w = image_shape
    center = (w // 2, h // 2)
    Y, X = np.ogrid[:h, :w]
    dist = (X - center[0]) ** 2 + (Y - center[1]) ** 2
    return dist <= radius**2
