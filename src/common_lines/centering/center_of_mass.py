import numpy as np

from ._validation import _as_single_image, _as_image_batch

from .shifts import shift_image
from .preprocessing import preprocess_images, PreprocessingOptions


def compute_center_of_mass(
    image: np.ndarray,
    eps: float = 1.0e-6,
    preprocessing_options: PreprocessingOptions | None = None,
) -> tuple[float, float]:
    """
    Calculates the center of mass of the given image. Returns it as a tuple with
    two elements ``(center_y, center_x)``.

    Parameters
    ----------
    image : np.ndarray
        Image to compute the center of mass for. Should be an array of shape ``(h, w)``.
    eps : float, optional
        Numerical tolerance to avoid division by zero, by default 1.0e-6
    preprocessing_options : PreprocessingOptions
        Parameters for ``preprocess_images``. See ``PreprocessingOptions``
        for more information.

    Returns
    -------
    tuple[float, float]
        Coordinates of the center of mass of the image, in ``(y, x)`` order.

    Raises
    ------
    ValueError
        If the total mass of any of the images is close to zero.
    """
    eps = _validate_eps(eps)
    image = _as_single_image(image)

    processed = preprocess_images(image[None], options=preprocessing_options)[0]

    # Varying axis 1 (columns) gives weight for each row
    row_weights = processed.sum(axis=1, dtype=np.float64)
    col_weights = processed.sum(axis=0, dtype=np.float64)

    total_weights = row_weights.sum(dtype=np.float64)

    if abs(total_weights) < eps:
        raise ValueError("image total intensity is too close to zero")

    weighted_y = np.dot(row_weights, np.arange(processed.shape[0], dtype=np.float64))
    weighted_x = np.dot(col_weights, np.arange(processed.shape[1], dtype=np.float64))

    return (float(weighted_y / total_weights), float(weighted_x / total_weights))


def compute_center_of_mass_batch(
    images: np.ndarray,
    *,
    eps: float = 1.0e-6,
    preprocessing_options: PreprocessingOptions | None = None,
) -> np.ndarray:
    """
    Computes the center of mass for each of the images in a batch. Returns an
    array of shape ``(batch, 2)`` containing the coordinates for these centers.

    Parameters
    ----------
    images : np.ndarray
        Image batch, array of shape ``(batch, height, width)``.
    eps : float, optional
        Numerical tolerance to avoid division by zero, by default 1.0e-6
    preprocessing_options : PreprocessingOptions
        Parameters for ``preprocess_images``. See ``PreprocessingOptions``
        for more information.

    Returns
    -------
    np.ndarray
        Coordinates for the center of mass of each image, array of shape
        ``(batch, 2)``. ``centers[i, 0]`` is the y coordinate for the center of
        mass of image ``images[i]``, while ``centers[i, 1]`` is its x coordinate.

    Raises
    ------
    ValueError
        If the total mass of any of the images is close to zero.
    """
    eps = _validate_eps(eps)
    images = _as_image_batch(images)

    processed = preprocess_images(images, options=preprocessing_options)

    batch, height, width = processed.shape

    # first coordinate is batch, second is y, third is x
    row_weights = processed.sum(axis=-1, dtype=np.float64)
    col_weights = processed.sum(axis=-2, dtype=np.float64)
    total_weights = row_weights.sum(axis=-1, dtype=np.float64)

    if np.any(np.abs(total_weights) < eps):
        raise ValueError(
            "At least one image has a total intensity that is too close to zero"
        )

    # row_weights is (batch, y_shape); (batch, y_shape) @ (y_shape, ) -> (batch, )
    weighted_y = row_weights @ np.arange(height, dtype=np.float64)
    weighted_x = col_weights @ np.arange(width, dtype=np.float64)

    centers = np.empty(shape=(batch, 2), dtype=np.float64)

    centers[:, 0] = weighted_y / total_weights
    centers[:, 1] = weighted_x / total_weights

    return centers


def estimate_center_of_mass_shift(
    image: np.ndarray,
    *,
    eps: float = 1.0e-6,
    preprocessing_options: PreprocessingOptions | None = None,
) -> tuple[float, float]:
    image = _as_single_image(image)
    height, width = image.shape

    center_y, center_x = compute_center_of_mass(
        image, eps=eps, preprocessing_options=preprocessing_options
    )

    return (height // 2 - center_y, width // 2 - center_x)


def estimate_center_of_mass_shifts(
    images: np.ndarray,
    *,
    eps: float = 1.0e-6,
    preprocessing_options: PreprocessingOptions | None = None,
) -> np.ndarray:
    images = _as_image_batch(images)

    centers = compute_center_of_mass_batch(
        images, eps=eps, preprocessing_options=preprocessing_options
    )
    geometric_center = np.array([images.shape[1] // 2, images.shape[2] // 2])

    return geometric_center - centers


def precenter_image_center_of_mass(
    image: np.ndarray,
    eps: float = 1.0e-6,
    preprocessing_options: PreprocessingOptions | None = None,
) -> np.ndarray:
    image = _as_single_image(image)
    shift_y, shift_x = estimate_center_of_mass_shift(
        image, eps=eps, preprocessing_options=preprocessing_options
    )
    return shift_image(image, shift_y, shift_x)


def _validate_eps(eps: float) -> float:
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite non-negative value.")
    return float(eps)
