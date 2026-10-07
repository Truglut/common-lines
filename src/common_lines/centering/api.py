from __future__ import annotations

from typing import Literal

import numpy as np

from .center_of_mass import estimate_center_of_mass_shifts
from .reflection import estimate_reflection_shifts
from .shifts import shift_images
from .preprocessing import PreprocessingOptions, create_circular_mask

PrecenteringMethod = Literal["reflection", "center_of_mass"]


def estimate_shifts(
    images: np.ndarray,
    *,
    method: PrecenteringMethod = "center_of_mass",
    preprocessing_options: PreprocessingOptions | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Estimate the shifts required to approximately center an image batch.

    When ``preprocessing_options`` is not supplied, an inscribed circular mask is used,
    the background is estimated from the region outside that mask, and negative values
    are left unchanged. Pass ``PreprocessingOptions()`` explicitly to disable this
    preprocessing.
    """
    if preprocessing_options is None:
        preprocessing_options = _default_center_of_mass_preprocessing(images)

    if method == "reflection":
        return estimate_reflection_shifts(
            images,
            preprocessing_options=preprocessing_options,
            **kwargs,
        )

    if method == "center_of_mass":
        return estimate_center_of_mass_shifts(
            images,
            preprocessing_options=preprocessing_options,
            **kwargs,
        )

    raise ValueError(f"Unknown centering method {method!r}.")


def precenter_images(
    images: np.ndarray,
    *,
    method: PrecenteringMethod = "center_of_mass",
    preprocessing_options: PreprocessingOptions | None = None,
    **kwargs,
) -> np.ndarray:
    """Approximately center an image batch and return shifted copies."""
    shifts = estimate_shifts(
        images,
        method=method,
        preprocessing_options=preprocessing_options,
        **kwargs,
    )
    return shift_images(images, shifts)


def _default_center_of_mass_preprocessing(images: np.ndarray) -> PreprocessingOptions:
    """Build the default preprocessing used by center-of-mass precentering."""
    image_shape = np.asarray(images).shape[-2:]
    radius = min(image_shape) / 2.0
    mask = create_circular_mask(image_shape, radius=radius)

    return PreprocessingOptions(
        mask=mask,
        subtract_background=True,
        clip_negative=False,
    )
