from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import numpy as np

from .center_of_mass import estimate_center_of_mass_shifts
from .reflection import estimate_reflection_shifts
from .shifts import shift_images

if TYPE_CHECKING:
    from common_lines.preprocessing import PreprocessingOptions

PrecenteringMethod = Literal["reflection", "center_of_mass"]


def estimate_shifts(
    images: np.ndarray,
    *,
    method: PrecenteringMethod = "center_of_mass",
    preprocessing_options: PreprocessingOptions | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Estimate translations that approximately center an image batch.

    By default, the images are preprocessed with an inscribed circular mask
    and background subtraction before the shifts are estimated. The same
    preprocessing default is used for both ``"center_of_mass"`` and
    ``"reflection"``.

    Pass ``PreprocessingOptions()`` explicitly to estimate shifts from the
    unprocessed images.

    Parameters
    ----------
    images
        Image batch with shape ``(n_images, height, width)``.
    method
        Centering method. Either ``"center_of_mass"`` or ``"reflection"``.
    preprocessing_options
        Preprocessing used only for estimating the shifts. The resulting
        shifts are applied to the original input images.
    """
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
    """
    Estimate particle shifts and apply them to the original images.

    ``preprocessing_options`` affects only the images used to estimate the
    shifts. Once the shifts have been estimated, they are applied to the
    original input images.

    Parameters
    ----------
    images
        Image batch with shape ``(n_images, height, width)``.
    method
        Method used to estimate the translations.
    preprocessing_options
        Preprocessing used when estimating the shifts.

    Returns
    -------
    numpy.ndarray
        Shifted copies of the original images.
    """
    shifts = estimate_shifts(
        images,
        method=method,
        preprocessing_options=preprocessing_options,
        **kwargs,
    )
    return shift_images(images, shifts)
