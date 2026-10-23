"""Shared CLI image preparation for the common-line validation benchmarks.

Use the same centering/preprocessing functions and ordering as
``common_lines.pipeline``. Both stages are disabled unless requested.
"""

import argparse

import numpy as np

from common_lines.centering import estimate_shifts, shift_images
from common_lines.preprocessing import (
    PreprocessingOptions,
    create_circular_mask,
    preprocess_images,
)


def add_preprocessing_arguments(parser: argparse.ArgumentParser):
    """Expose the image-preparation settings of the public pipeline."""
    parser.add_argument(
        "--precenter",
        choices=("none", "center_of_mass", "reflection"),
        default="none",
        help="Precentering method (default: none)",
    )
    parser.add_argument(
        "--preprocessing",
        choices=("none", "default"),
        default="none",
        help="Preprocessing preset (default: none). 'default' enables the "
        "circular mask and background subtraction.",
    )
    parser.add_argument(
        "--mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override circular masking in the selected preprocessing preset",
    )
    parser.add_argument(
        "--subtract-background",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override background subtraction (estimated outside a circular mask)",
    )
    parser.add_argument(
        "--clip-negative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override clipping negative pixel values to zero",
    )
    parser.add_argument(
        "--mask-radius-fraction",
        type=float,
        default=1.0,
        help="Circular-mask radius relative to half the image size (default: 1)",
    )
    parser.add_argument(
        "--gaussian-lowpass-sigma",
        type=float,
        default=None,
        metavar="PIXELS",
        help="Enable Gaussian low-pass filtering with this positive spatial sigma "
        "(pixels; default: disabled). Can be used with --preprocessing none.",
    )


def resolve_preprocessing_options(images, args):
    """Translate CLI controls to the current ``PreprocessingOptions`` API.

    Return None when every step is disabled. Background estimation may use an
    auxiliary circular mask even when masking the signal is disabled.
    """
    radius_fraction = args.mask_radius_fraction
    if not np.isfinite(radius_fraction) or not (0 < radius_fraction <= 1):
        raise ValueError("--mask-radius-fraction must be finite and in (0, 1]")

    sigma = args.gaussian_lowpass_sigma
    if sigma is not None and (not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("--gaussian-lowpass-sigma must be finite and positive")

    use_defaults = args.preprocessing == "default"
    mask_enabled = use_defaults if args.mask is None else args.mask
    background_enabled = (
        use_defaults if args.subtract_background is None else args.subtract_background
    )
    clip_enabled = False if args.clip_negative is None else args.clip_negative

    if not (mask_enabled or background_enabled or clip_enabled or sigma is not None):
        return None

    circle = None
    if mask_enabled or background_enabled:
        shape = np.asarray(images).shape[-2:]
        circle = create_circular_mask(shape, min(shape) / 2 * radius_fraction)
    return PreprocessingOptions(
        mask=circle if mask_enabled else None,
        subtract_background=background_enabled,
        background_mask=(
            1.0 - circle if background_enabled and not mask_enabled else None
        ),
        clip_negative=clip_enabled,
        gaussian_lowpass_sigma=sigma,
    )


def prepare_images(images, *, precenter, preprocessing_options):
    """Replicate the public pipeline: preprocess for shifts, shift raw images,
    then preprocess the shifted images for common-line detection.
    """
    if precenter != "none":
        images_for_centering = preprocess_images(images, options=preprocessing_options)
        shifts = estimate_shifts(images_for_centering, method=precenter)
        images = shift_images(images, shifts)
    return preprocess_images(images, options=preprocessing_options)


def configuration_description(args):
    def resolve(value, default):
        return default if value is None else value

    preset = args.preprocessing == "default"
    mask = resolve(args.mask, preset)
    background = resolve(args.subtract_background, preset)
    clip = resolve(args.clip_negative, False)
    return (
        f"precenter={args.precenter}, preprocessing={args.preprocessing}, "
        f"mask={mask}, subtract_background={background}, clip_negative={clip}, "
        f"mask_radius_fraction={args.mask_radius_fraction:g}, "
        f"gaussian_lowpass_sigma={args.gaussian_lowpass_sigma}"
    )
