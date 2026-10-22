from __future__ import annotations

from typing import Literal

import numpy as np

from .center_of_mass import estimate_center_of_mass_shifts
from .reflection import estimate_reflection_shifts
from .shifts import shift_images

PrecenteringMethod = Literal["reflection", "center_of_mass"]


def estimate_shifts(
    images: np.ndarray,
    *,
    method: PrecenteringMethod = "center_of_mass",
    **kwargs,
) -> np.ndarray:
    """
    Estimate translations that approximately center an image batch.

    Parameters
    ----------
    images
        Image batch with shape ``(n_images, height, width)``.
        Any preprocessing must be performed by the caller.
    method
        Centering method: ``"center_of_mass"`` or ``"reflection"``.
    **kwargs
        Additional options passed to the selected shift estimator.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_images, 2)`` containing ``(shift_y, shift_x)``.
    """
    if method == "reflection":
        return estimate_reflection_shifts(images, **kwargs)

    if method == "center_of_mass":
        return estimate_center_of_mass_shifts(images, **kwargs)

    raise ValueError(f"Unknown centering method {method!r}.")


def precenter_images(
    images: np.ndarray,
    *,
    method: PrecenteringMethod = "center_of_mass",
    **kwargs,
) -> np.ndarray:
    """
    Estimate centering shifts and apply them to the supplied images.

    Any preprocessing must be performed by the caller. The same supplied
    images are used for shift estimation and translation.

    Parameters
    ----------
    images
        Image batch with shape ``(n_images, height, width)``.
    method
        Method used to estimate the translations.
    **kwargs
        Additional options passed to the selected shift estimator.

    Returns
    -------
    numpy.ndarray
        Shifted copies of the supplied images.
    """
    shifts = estimate_shifts(images, method=method, **kwargs)
    return shift_images(images, shifts)
