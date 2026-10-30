"""
Benchmark the complete relative-orientation graph with symmetry-aware scoring.

Runs the PUBLIC graph pipeline on selected experimental or synthetic images,
then compares every directed edge with ground-truth relative rotations.
Precentering, preprocessing, detection, triplet selection/aggregation, fallback,
and graph-level handedness selection all run inside the existing pipeline.

Examples (from repository root):
    python -m scripts.pipeline_errors particles.npz --n-images 40 --angle-step 1
    python -m scripts.pipeline_errors particles.npz --n-third 5 --precenter none
    python -m scripts.pipeline_errors particles.npz --preprocessing default --n-third 10
    python -m scripts.pipeline_errors particles.npz --n-third 24 --reconstitution-method averaging
    python -m scripts.pipeline_errors particles.star --image-stack particles.mrcs

Ground-truth conventions: R_i maps reference coordinates into image i;
for directed edge i -> j, the true rotation is R_j @ R_i.T.

The two *global* handedness choices (estimated graph, J @ graph @ J) are
compared against ground truth. One branch is used for ALL directed edges;
handedness is never chosen separately for individual edges. The handedness
selected for scoring is an oracle used only in this validation script.

The saved NPZ graph is the ORIGINAL pipeline output, not the oracle-mirrored
version. Fallback edges are included in graph accuracy but also reported
separately: a fallback rotation does not fully determine the 3D orientation.
C1, Cn and Dn are supported as EVALUATION symmetry groups. For each directed
edge, the reported angular error is minimized over all equivalent true relative
rotations R_j @ S @ R_i.T. This is an edge-wise symmetry oracle: different edges
can choose incompatible symmetry operators, so it does NOT measure whether
estimated rotations admit one globally consistent symmetry assignment.
The underlying PUBLIC graph estimator remains unchanged and is not itself
symmetry-aware.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common_lines import (
    build_relative_orientations_graph_with_diagnostics,
    mirror_relative_orientations_graph,
)

if __package__:
    from ._benchmark_preprocessing import (
        add_preprocessing_arguments,
        configuration_description,
        resolve_preprocessing_options,
    )
    from .detection_errors import read_images, read_metadata, symmetry_operators
else:
    from _benchmark_preprocessing import (
        add_preprocessing_arguments,
        configuration_description,
        resolve_preprocessing_options,
    )
    from detection_errors import read_images, read_metadata, symmetry_operators


J = np.diag([1.0, 1.0, -1.0])
CSV_COLUMNS = (
    "particle_i",
    "particle_j",
    "used_fallback",
    "valid_rotation",
    "rotation_error_deg",
    "inverse_disagreement_deg",
    "orthogonality_residual",
    "determinant",
    "symmetry_operator_index",
)


def rotation_error_deg(estimated, true):
    """Geodesic SO(3) distance in degrees; vectorized over leading axes."""
    dot = np.einsum("...ab,...ab->...", estimated, true)
    return np.rad2deg(np.arccos(np.clip((dot - 1.0) / 2.0, -1.0, 1.0)))


def valid_rotation_mask(graph, atol=1e-5):
    """Mask for finite near-SO(3) matrices; also return raw diagnostics."""
    with np.errstate(invalid="ignore", over="ignore"):
        residual = np.linalg.norm(
            np.swapaxes(graph, -1, -2) @ graph - np.eye(3),
            axis=(-2, -1),
        )
        determinant = np.linalg.det(graph)
    valid = (
        np.isfinite(graph).all(axis=(-2, -1))
        & np.isfinite(residual)
        & np.isfinite(determinant)
        & (residual <= atol)
        & (np.abs(determinant - 1.0) <= atol)
    )
    return valid, residual, determinant


def evaluate_graph(graph, used_fallback, matrices, symmetry_ops=None):
    """Score one graph-wide handedness; minimize symmetry per directed edge.

    True relative rotations are R_j @ S @ R_i.T (reference->image convention).
    Minimizing S independently for each edge is an *oracle* quotient-SO(3)
    metric, not a test of global symmetry-gauge consistency. Invalid output
    rotations remain invalid; they are not silently projected to SO(3).
    """
    if symmetry_ops is None:
        symmetry_ops = np.eye(3)[None, ...]
    n = len(matrices)
    graph = np.asarray(graph, dtype=float)
    used_fallback = np.asarray(used_fallback, dtype=bool)
    if graph.shape != (n, n, 3, 3) or used_fallback.shape != (n, n):
        raise ValueError("Graph/fallback shapes do not match the selected orientations")

    valid, residual, det = valid_rotation_mask(graph)
    offdiag = ~np.eye(n, dtype=bool)
    eligible = valid & offdiag
    if not np.any(eligible):
        raise ValueError(
            "Pipeline did not return any valid directed relative rotations"
        )

    mirrored = mirror_relative_orientations_graph(graph)
    inverse_i = np.swapaxes(matrices[:, None, :, :], -1, -2)
    # Retain the minimizing symmetry index for each directed edge, separately
    # for the two graph-wide handedness branches.
    direct_errors = np.full((n, n), np.inf)
    mirrored_errors = np.full((n, n), np.inf)
    direct_selected = np.full((n, n), -1, dtype=int)
    mirrored_selected = np.full((n, n), -1, dtype=int)
    for k, operator in enumerate(symmetry_ops):
        # truth[i,j] = R_j @ S @ R_i.T, NOT S @ R_j @ R_i.T.
        true_rotations = (matrices @ operator)[None, :, :, :] @ inverse_i
        with np.errstate(invalid="ignore"):
            candidate_direct = rotation_error_deg(graph, true_rotations)
            candidate_mirror = rotation_error_deg(mirrored, true_rotations)
        better_direct = candidate_direct < direct_errors
        better_mirror = candidate_mirror < mirrored_errors
        direct_errors[better_direct] = candidate_direct[better_direct]
        mirrored_errors[better_mirror] = candidate_mirror[better_mirror]
        direct_selected[better_direct] = k
        mirrored_selected[better_mirror] = k
    direct_errors[~eligible] = np.nan
    mirrored_errors[~eligible] = np.nan

    # IMPORTANT: one global handedness branch, NOT a per-edge hand oracle.
    use_mirror = bool(np.nanmean(mirrored_errors) < np.nanmean(direct_errors))
    errors = mirrored_errors if use_mirror else direct_errors
    symmetry_selected = mirrored_selected if use_mirror else direct_selected
    symmetry_selected = symmetry_selected.copy()
    symmetry_selected[~eligible] = -1
    chosen = mirrored if use_mirror else graph

    # Independent estimates of i->j and j->i need not be inverses.
    with np.errstate(invalid="ignore"):
        inverse_errors = rotation_error_deg(
            chosen, np.swapaxes(chosen.transpose(1, 0, 2, 3), -1, -2)
        )
    inverse_errors[~(eligible & eligible.T)] = np.nan
    return {
        "errors": errors,
        "valid": valid,
        "residual": residual,
        "determinant": det,
        "inverse_disagreement": inverse_errors,
        "used_mirrored_branch": use_mirror,
        "symmetry_operator_index": symmetry_selected,
        "used_fallback": used_fallback,
    }


def largest_component_size(used_fallback, valid):
    """Size of largest weakly connected component of non-fallback edges."""
    usable = valid & ~used_fallback
    adjacency = usable | usable.T
    np.fill_diagonal(adjacency, False)
    unvisited = set(range(len(adjacency)))
    largest = 0
    while unvisited:
        start = unvisited.pop()
        stack = [start]
        size = 1
        while stack:
            node = stack.pop()
            neighbors = set(np.flatnonzero(adjacency[node])) & unvisited
            unvisited.difference_update(neighbors)
            stack.extend(neighbors)
            size += len(neighbors)
        largest = max(largest, size)
    return largest


def _accuracy_summary(errors, denominator):
    """Report rates among valid edges and among every requested directed edge."""
    finite = np.isfinite(errors)
    e = errors[finite]
    if not len(e):
        return None
    return {
        "count": len(e),
        "mean": float(e.mean()),
        "median": float(np.median(e)),
        "within10": float(np.mean(e <= 10)),
        "within20": float(np.mean(e <= 20)),
        "total_within20": float(np.count_nonzero(e <= 20) / denominator),
    }


def print_summary(result, n_images, handedness_reference):
    errors = result["errors"]
    fallback = result["used_fallback"]
    valid = result["valid"]
    offdiag = ~np.eye(n_images, dtype=bool)
    total = int(offdiag.sum())
    nonfallback = offdiag & ~fallback
    only_fallback = offdiag & fallback

    print(f"Images: {n_images}; directed edges: {total} ({total // 2} unordered pairs)")
    print(
        f"Handedness reference (selected-image indices, 0-based): {handedness_reference}"
    )
    print(
        "Evaluation handedness: "
        + ("mirrored graph" if result["used_mirrored_branch"] else "as returned")
    )
    print(
        f"Non-fallback directed edges: {nonfallback.sum()} ({100 * nonfallback.sum() / total:.1f}%)"
    )
    print(
        f"Fallback directed edges:     {only_fallback.sum()} ({100 * only_fallback.sum() / total:.1f}%)"
    )
    print(f"Invalid directed rotations: {(offdiag & ~valid).sum()}")
    print(
        f"Largest non-fallback weak component: "
        f"{largest_component_size(fallback, valid)} / {n_images} images"
    )

    for label, selection in (
        ("All edges", offdiag),
        ("Non-fallback edges", nonfallback),
        ("Fallback edges", only_fallback),
    ):
        stats = (
            _accuracy_summary(errors[selection], int(selection.sum()))
            if selection.any()
            else None
        )
        print(f"\n{label}:")
        if stats is None:
            print("  No valid rotations")
            continue
        print(f"  Valid: {stats['count']} / {selection.sum()}")
        print(f"  Mean rotation error:       {stats['mean']:.2f} deg")
        print(f"  Median rotation error:     {stats['median']:.2f} deg")
        print(f"  Within 10 deg (valid):     {100 * stats['within10']:.1f}%")
        print(f"  Within 20 deg (valid):     {100 * stats['within20']:.1f}%")
        print(
            f"  Valid and within 20 deg:  {100 * stats['total_within20']:.1f}% of all edges in group"
        )

    i, j = np.triu_indices(n_images, k=1)
    inverse = result["inverse_disagreement"][i, j]
    if np.any(np.isfinite(inverse)):
        print(
            f"\nMedian opposite-direction disagreement: {np.nanmedian(inverse):.2f} deg"
        )


def save_results(
    prefix, selected_indices, graph, result, handedness_reference, configuration=None
):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    n = len(selected_indices)
    i, j = np.nonzero(~np.eye(n, dtype=bool))
    with prefix.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for a, b in zip(i, j):
            writer.writerow(
                (
                    int(selected_indices[a] + 1),
                    int(selected_indices[b] + 1),
                    int(result["used_fallback"][a, b]),
                    int(result["valid"][a, b]),
                    float(result["errors"][a, b]),
                    float(result["inverse_disagreement"][a, b]),
                    float(result["residual"][a, b]),
                    float(result["determinant"][a, b]),
                    int(result["symmetry_operator_index"][a, b]),
                )
            )

    # The stored graph is the ORIGINAL estimate, not the oracle mirrored variant.
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        graph=graph,
        used_fallback=result["used_fallback"],
        selected_indices=selected_indices,
        handedness_reference=np.asarray(handedness_reference, dtype=int),
        oracle_mirrored_for_scoring=np.asarray(result["used_mirrored_branch"]),
        symmetry_operator_index=result["symmetry_operator_index"],
        run_configuration_json=np.asarray(
            json.dumps(configuration or {}, default=str, sort_keys=True)
        ),
    )

    errors = result["errors"]
    fallback = result["used_fallback"]
    offdiag = ~np.eye(n, dtype=bool)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(0, 180, 37)
    for label, subset in (
        ("Non-fallback", offdiag & ~fallback),
        ("Fallback", offdiag & fallback),
    ):
        values = errors[subset]
        values = values[np.isfinite(values)]
        if len(values):
            axes[0].hist(
                values,
                bins=bins,
                histtype="step",
                linewidth=1.8,
                weights=np.full(len(values), 1.0 / len(values)),
                label=f"{label} (n={len(values)})",
            )
    axes[0].set(
        title="Relative-rotation errors",
        xlabel="Directed edge error (deg)",
        ylabel="Fraction of edges in each group per bin",
    )
    if len(axes[0].lines) or len(axes[0].patches):
        axes[0].legend(fontsize=8)

    i, j = np.triu_indices(n, 1)
    inverse = result["inverse_disagreement"][i, j]
    average_error = (errors[i, j] + errors[j, i]) / 2
    eligible = np.isfinite(inverse) & np.isfinite(average_error)
    for label, mask in (
        ("Both directions non-fallback", ~fallback[i, j] & ~fallback[j, i]),
        ("At least one fallback", fallback[i, j] | fallback[j, i]),
    ):
        selected = eligible & mask
        if np.any(selected):
            axes[1].scatter(
                inverse[selected],
                average_error[selected],
                s=11,
                alpha=0.45,
                label=label,
            )
    axes[1].set(
        title="Inverse consistency versus error",
        xlabel="Disagreement between i→j and inverse(j→i) (deg)",
        ylabel="Mean error of both directions (deg)",
    )
    if axes[1].collections:
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=160)
    plt.close(fig)

    print(
        f"\nSaved: {prefix.with_suffix('.csv')}, {prefix.with_suffix('.png')}, "
        f"{prefix.with_suffix('.npz')}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "metadata", type=Path, help="NPZ or STAR ground-truth orientation metadata"
    )
    parser.add_argument(
        "--image-stack",
        type=Path,
        help="Optional MRC/MRCS image stack in metadata order",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=24,
        help="Randomly select up to this many images (default: 24)",
    )
    parser.add_argument(
        "--angle-step",
        type=float,
        default=2.0,
        help="Sinogram angular spacing in degrees (default: 2)",
    )
    parser.add_argument(
        "--symmetry",
        default="C1",
        help="Reference-object symmetry, Cn or Dn (default: C1)",
    )
    parser.add_argument(
        "--n-third",
        type=int,
        default=1,
        help="Maximum valid third-image witnesses per directed edge (default: 1)",
    )
    parser.add_argument("--common-line-block-size", type=int, default=4)
    parser.add_argument(
        "--reconstitution-method",
        choices=("averaging", "voting"),
        default="voting",
        help="Triplet aggregation method (default: voting)",
    )
    parser.add_argument(
        "--handedness-reference",
        type=int,
        nargs=2,
        metavar=("I", "J"),
        help="Optional selected-image indices (0-based) for pipeline handedness reference",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Sampling and pipeline seed"
    )
    parser.add_argument("--output", type=Path, default=Path("graph_validation"))
    add_preprocessing_arguments(parser)
    args = parser.parse_args()
    operators = symmetry_operators(args.symmetry)

    if args.n_images < 3 or not np.isfinite(args.angle_step) or args.angle_step <= 0:
        parser.error(
            "--n-images must be >= 3 and --angle-step must be finite and positive"
        )
    if args.n_third < 1 or args.common_line_block_size < 1:
        parser.error("--n-third and --common-line-block-size must be >= 1")

    matrices, names, embedded = read_metadata(args.metadata)
    if matrices.ndim != 3 or matrices.shape[-2:] != (3, 3) or len(matrices) < 3:
        raise ValueError(
            "Expected at least three reference-to-image rotation matrices (N,3,3)"
        )
    rng = np.random.default_rng(args.seed)
    indices = np.sort(
        rng.choice(len(matrices), size=min(len(matrices), args.n_images), replace=False)
    )
    selected_matrices = matrices[indices]
    images = read_images(args.metadata, indices, names, embedded, args.image_stack)

    if args.n_third > len(images) - 2:
        parser.error(f"--n-third cannot exceed n-images - 2 = {len(images) - 2}")
    reference = tuple(args.handedness_reference) if args.handedness_reference else None
    options = resolve_preprocessing_options(images, args)
    theta = np.arange(0.0, 180.0, args.angle_step)

    print(f"Symmetry: {args.symmetry.upper()} (edge-wise symmetry-aware scoring)")
    if len(operators) > 1:
        print("Note: this score does not enforce graph-wide symmetry consistency.")
    print(f"Image preparation: {configuration_description(args)}")
    print(f"n_third={args.n_third}, angle_step={args.angle_step:g}, seed={args.seed}")
    print(f"Reconstitution method: {args.reconstitution_method}")
    estimate = build_relative_orientations_graph_with_diagnostics(
        images,
        n_third=args.n_third,
        random_seed=args.seed,
        handedness_reference=reference,
        theta=theta,
        precenter_method=None if args.precenter == "none" else args.precenter,
        common_line_block_size=args.common_line_block_size,
        preprocessing_options=options,
        reconstitution_method=args.reconstitution_method,
    )
    result = evaluate_graph(
        estimate.graph,
        estimate.used_fallback,
        selected_matrices,
        symmetry_ops=operators,
    )
    print_summary(result, len(images), estimate.handedness_reference)
    save_results(
        args.output,
        indices,
        estimate.graph,
        result,
        estimate.handedness_reference,
        configuration=vars(args),
    )


if __name__ == "__main__":
    main()
