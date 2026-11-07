"""Shared soft-voting controls for the orientation benchmark CLIs."""

import argparse

import numpy as np

from common_lines import SoftVotingOptions


def add_soft_voting_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose optional overrides, keeping defaults in SoftVotingOptions."""
    defaults = SoftVotingOptions()
    group = parser.add_argument_group(
        "soft voting (requires --reconstitution-method soft_voting)"
    )
    group.add_argument(
        "--soft-voting-bins",
        type=int,
        help=f"Initial grid points on [0, pi] (default: {defaults.voting_bins})",
    )
    group.add_argument(
        "--soft-voting-sigma-deg",
        type=float,
        help=f"Gaussian bandwidth in degrees (default: {np.rad2deg(defaults.voting_sigma):g})",
    )
    group.add_argument(
        "--soft-voting-max-its",
        type=int,
        help=f"Maximum reliability updates (default: {defaults.max_its})",
    )
    group.add_argument(
        "--soft-voting-stopping-tol",
        type=float,
        help=f"Relative reliability tolerance; zero disables early stopping (default: {defaults.stopping_tol:g})",
    )


def resolve_soft_voting_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> SoftVotingOptions | None:
    """Validate CLI overrides and convert the bandwidth from degrees to radians.

    Return None for other methods; explicit overrides require soft voting.
    Resolve defaults even when no overrides were supplied, so saved run
    configurations can record the settings actually used.
    """
    overrides = (
        args.soft_voting_bins,
        args.soft_voting_sigma_deg,
        args.soft_voting_max_its,
        args.soft_voting_stopping_tol,
    )
    if args.reconstitution_method != "soft_voting":
        if any(value is not None for value in overrides):
            parser.error(
                "--soft-voting-* options require --reconstitution-method soft_voting"
            )
        return None

    options = SoftVotingOptions()
    if args.soft_voting_bins is not None:
        options.voting_bins = args.soft_voting_bins
    if args.soft_voting_sigma_deg is not None:
        options.voting_sigma = float(np.deg2rad(args.soft_voting_sigma_deg))
    if args.soft_voting_max_its is not None:
        options.max_its = args.soft_voting_max_its
    if args.soft_voting_stopping_tol is not None:
        options.stopping_tol = args.soft_voting_stopping_tol

    if options.voting_bins < 2:
        parser.error("--soft-voting-bins must be at least 2")
    if not np.isfinite(options.voting_sigma) or options.voting_sigma <= 0:
        parser.error("--soft-voting-sigma-deg must be finite and positive")
    if options.max_its < 1:
        parser.error("--soft-voting-max-its must be positive")
    if not np.isfinite(options.stopping_tol) or options.stopping_tol < 0:
        parser.error("--soft-voting-stopping-tol must be finite and nonnegative")
    return options
