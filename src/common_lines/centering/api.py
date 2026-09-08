from __future__ import annotations

from typing import Literal

import numpy as np

from .center_of_mass import estimate_center_of_mass_shifts
from .reflection import estimate_reflection_shifts
from .shifts import shift_images

PrecenteringMethod = Literal["reflection", "center_of_mass"]


def estimate_shifts(
    images: np.ndarray, *, method: PrecenteringMethod = "center_of_mass", **kwargs
) -> np.ndarray:
    if method == "reflection":
        return estimate_reflection_shifts(images, **kwargs)

    if method == "center_of_mass":
        return estimate_center_of_mass_shifts(images, **kwargs)

    raise ValueError(f"Unknown centering method {method!r}.")


def precenter_images(
    images: np.ndarray, *, method: PrecenteringMethod = "center_of_mass", **kwargs
) -> np.ndarray:
    shifts = estimate_shifts(images, method=method, **kwargs)
    return shift_images(images, shifts)
