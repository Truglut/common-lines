"""CLI image-preparation options shared by the two validation benchmarks.

These call the same centering/preprocessing functions and use the same ordering
as ``common_lines.main``. They do not implement any new image processing.
"""

import argparse

import numpy as np

from common_lines.centering import precenter_images
from common_lines.centering.preprocessing import (
    PreprocessingOptions,
    create_circular_mask,
    preprocess_images,
)


def add_preprocessing_arguments(
    parser: argparse.ArgumentParser, *, pipeline_defaults: bool
):
    """Expose the main pipeline's existing centering and preprocessing options."""
    parser.add_argument(
        "--precenter",
        choices=("none", "center_of_mass", "reflection"),
        default="center_of_mass" if pipeline_defaults else "none",
        help="Precentering method (default: %(default)s)",
    )
    parser.add_argument(
        "--preprocessing",
        choices=("none", "default"),
        default="default" if pipeline_defaults else "none",
        help="Image preprocessing preset (default: %(default)s). 'default' is "
        "the pipeline's circular mask and background subtraction.",
    )
    parser.add_argument(
        "--mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether a centered circular mask is applied",
    )
    parser.add_argument(
        "--subtract-background",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override background subtraction (estimated outside the circular mask)",
    )
    parser.add_argument(
        "--clip-negative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override clipping negative image values to zero",
    )
    parser.add_argument(
        "--mask-radius-fraction",
        type=float,
        default=1.0,
        help="Circular mask radius as a fraction of half the image size (default: 1.0)",
    )


def resolve_preprocessing_options(images, args):
    """Resolve the requested options using the existing PreprocessingOptions API.

    Background subtraction can be enabled without applying a signal mask: in
    that case the exterior of the same circular region estimates the background.
    The defaults for 'default' exactly match main.py's pipeline defaults.
    """
    radius_fraction = args.mask_radius_fraction
    if not np.isfinite(radius_fraction) or not (0 < radius_fraction <= 1):
        raise ValueError("--mask-radius-fraction must be in (0, 1]")

    use_defaults = args.preprocessing == "default"
    mask_enabled = use_defaults if args.mask is None else args.mask
    background_enabled = (
        use_defaults if args.subtract_background is None else args.subtract_background
    )
    clip_enabled = False if args.clip_negative is None else args.clip_negative

    if not (mask_enabled or background_enabled or clip_enabled):
        return PreprocessingOptions()

    shape = np.asarray(images).shape[-2:]
    circle = create_circular_mask(shape, min(shape) / 2 * radius_fraction)
    return PreprocessingOptions(
        mask=circle if mask_enabled else None,
        subtract_background=background_enabled,
        background_mask=(
            (1.0 - circle) if background_enabled and not mask_enabled else None
        ),
        clip_negative=clip_enabled,
    )


def prepare_images(images, *, precenter, preprocessing_options):
    """Replicate main.py's order: precenter originals, then preprocess copies."""
    if precenter != "none":
        images = precenter_images(
            images,
            method=precenter,
            preprocessing_options=preprocessing_options,
        )
    return preprocess_images(images, options=preprocessing_options)


def configuration_description(args):
    def resolve(value, default):
        return default if value is None else value

    pipeline_default = args.preprocessing == "default"
    mask = resolve(args.mask, pipeline_default)
    background = resolve(args.subtract_background, pipeline_default)
    clip = resolve(args.clip_negative, False)
    return (
        f"precenter={args.precenter}, preprocessing={args.preprocessing}, "
        f"mask={mask}, subtract_background={background}, clip_negative={clip}, "
        f"mask_radius_fraction={args.mask_radius_fraction:g}"
    )
