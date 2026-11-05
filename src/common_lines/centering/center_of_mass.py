import numpy as np

from ._validation import _as_image_batch, _as_single_image
from .shifts import shift_image


def compute_center_of_mass(
    image: np.ndarray, eps: float = 1.0e-6
) -> tuple[float, float]:
    """
    Calculates the center of mass of the given image. Returns it as a tuple with
    two elements ``(center_y, center_x)``.

    The supplied image values are used directly; any preprocessing must be performed
    by the caller.

    Parameters
    ----------
    image : np.ndarray
        Image to compute the center of mass for. Should be an array of shape ``(h, w)``.
    eps : float, optional
        Numerical tolerance to avoid division by zero, by default 1.0e-6

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

    # Varying axis 1 (columns) gives weight for each row
    row_weights = image.sum(axis=1, dtype=np.float64)
    col_weights = image.sum(axis=0, dtype=np.float64)

    total_weights = row_weights.sum(dtype=np.float64)

    if abs(total_weights) < eps:
        raise ValueError("image total intensity is too close to zero")

    weighted_y = np.dot(row_weights, np.arange(image.shape[0], dtype=np.float64))
    weighted_x = np.dot(col_weights, np.arange(image.shape[1], dtype=np.float64))

    return (float(weighted_y / total_weights), float(weighted_x / total_weights))


def compute_center_of_mass_batch(
    images: np.ndarray, *, eps: float = 1.0e-6
) -> np.ndarray:
    """
    Computes the center of mass for each of the images in a batch. Returns an
    array of shape ``(batch, 2)`` containing the coordinates for these centers.

    The supplied image values are used directly; any preprocessing must be performed
    by the caller.

    Parameters
    ----------
    images : np.ndarray
        Image batch, array of shape ``(batch, height, width)``.
    eps : float, optional
        Numerical tolerance to avoid division by zero, by default 1.0e-6

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

    batch, height, width = images.shape

    # first coordinate is batch, second is y, third is x
    row_weights = images.sum(axis=-1, dtype=np.float64)
    col_weights = images.sum(axis=-2, dtype=np.float64)
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
    image: np.ndarray, *, eps: float = 1.0e-6
) -> tuple[float, float]:
    """
    Estimate the translation that moves an image's intensity center to its center pixel.

    Parameters
    ----------
    image : np.ndarray
        Real-valued array of shape ``(height, width)``. Intensities are used
        directly as weights; background subtraction and clipping, if needed,
        must be performed by the caller.
    eps : float, optional
        Threshold below which the absolute total intensity is rejected as
        too close to zero. Default is 1.0e-6.

    Returns
    -------
    shift_y, shift_x : float
        Corrections in pixels, computed as ``(height // 2, width // 2)`` minus
        the intensity center. Positive values move content towards increasing
        row or column indices. The target matches the Radon rotation center,
        including for even image sizes.

    Raises
    ------
    ValueError
        If the image or tolerance is invalid, or the total intensity is too
        close to zero.
    """
    image = _as_single_image(image)
    height, width = image.shape

    center_y, center_x = compute_center_of_mass(image, eps=eps)

    return (height // 2 - center_y, width // 2 - center_x)


def estimate_center_of_mass_shifts(
    images: np.ndarray, *, eps: float = 1.0e-6
) -> np.ndarray:
    """
    Estimate intensity-based centering corrections for an image batch.

    ``images`` has shape ``(batch, height, width)``. Return a float64 array of
    shape ``(batch, 2)`` containing ``(shift_y, shift_x)`` in pixels, targeting
    ``(height // 2, width // 2)``. The input is unchanged. Preprocessing and
    ``eps`` follow ``estimate_center_of_mass_shift``; an invalid or near-zero
    total intensity in any image raises ``ValueError`` for the whole batch.
    """
    images = _as_image_batch(images)

    centers = compute_center_of_mass_batch(images, eps=eps)
    geometric_center = np.array([images.shape[1] // 2, images.shape[2] // 2])

    return geometric_center - centers


def precenter_image_center_of_mass(
    image: np.ndarray, eps: float = 1.0e-6
) -> np.ndarray:
    """Return a centered copy of one ``(height, width)`` image.

    Estimate the correction with ``estimate_center_of_mass_shift`` using the
    supplied intensities and ``eps``. Apply it with bilinear interpolation and
    zero padding, as in ``shift_image``. No preprocessing is performed here.
    """
    image = _as_single_image(image)
    shift_y, shift_x = estimate_center_of_mass_shift(image, eps=eps)
    return shift_image(image, shift_y, shift_x)


def _validate_eps(eps: float) -> float:
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite non-negative value.")
    return float(eps)
