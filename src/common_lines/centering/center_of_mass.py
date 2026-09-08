import numpy as np

from ._validation import _as_single_image

from .shifts import shift_image


def compute_center_of_mass(
    image: np.ndarray, eps: float = 1.0e-6
) -> tuple[float, float]:
    image = _as_single_image(image)

    row_weights = image.sum(axis=1)  # the y coordinate varies with axis=0
    col_weights = image.sum(axis=0)  # the x coordinate varies with axis=1
    total_weights = row_weights.sum()

    if abs(total_weights) < eps:
        raise ValueError("image total intensity is too close to zero")

    weighted_y = np.dot(row_weights, np.arange(image.shape[0], dtype=np.float64))
    weighted_x = np.dot(col_weights, np.arange(image.shape[1], dtype=np.float64))

    return (float(weighted_y / total_weights), float(weighted_x / total_weights))


def estimate_center_of_mass_shift(image: np.ndarray) -> tuple[float, float]:
    center_y, center_x = compute_center_of_mass(image)

    height, width = image.shape

    return (height // 2 - center_y, width // 2 - center_x)


def precenter_image_center_of_mass(image: np.ndarray) -> np.ndarray:
    shift_y, shift_x = estimate_center_of_mass_shift(image)

    return shift_image(image, shift_y, shift_x)
