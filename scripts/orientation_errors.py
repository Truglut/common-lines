"""
Benchmark symmetry-aware relative rotations from Van Heel angular reconstitution.

Examples (from the repository root):
    python -m scripts.orientation_errors data/particles.npz --n-images 40
    python -m scripts.orientation_errors data/particles.npz --modes exact perturbed
    python -m scripts.orientation_errors data/particles.npz --noise-std 1 2 5 10 \\
        --max-triplets 1000 --angle-step 1

The three modes share the same sampled triplets:
  exact:       common lines from one consistent choice of ground-truth orientations
  perturbed:   ground-truth common-line angles plus independent Gaussian noise
  detected:    common lines measured from images using the project's detector

Each valid triplet yields three relative rotations. A single handedness branch is
chosen for the *whole triplet* using ground truth, purely for validation. This
oracle selection is NOT a deployable handedness estimator. The benchmark tests
local triplet reconstitution, not the full graph aggregation/handedness stage
of build_relative_orientations_graph(). Symmetry-aware validation minimizes over
CONSISTENT symmetry-equivalent orientations of all three triplet images, fixing
the symmetry gauge of the first image. This is an oracle scoring convention, not
a symmetry-aware angular-reconstitution method. In the exact/perturbed modes,
common lines are calculated from one consistent choice of orientations; detected
mode still uses the ordinary single-peak C1 common-line detector.

Outputs: <prefix>.csv (one row per attempted triplet and mode), <prefix>.png.
By default, detected mode uses raw images, matching detection_errors.py.
Use --precenter and --preprocessing (with optional --mask, --subtract-background,
--clip-negative, --gaussian-lowpass-sigma and --mask-radius-fraction overrides) to match the image
preparation options of detection_errors.py and pipeline_errors.py.
The older --pipeline-preprocessing flag remains as a shorthand for the formerly
default center-of-mass centering and mask/background preprocessing. Exact and perturbed modes use ground-truth
angles and do not process images. None of these modes runs pipeline.py's graph-level
postprocessing. For STAR metadata or MRC stacks, see detection_errors.py.
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common_lines.alignment import compute_transformation_matrices_from_angles
from common_lines.detection import detect_common_lines_all_pairs
from common_lines.forward import compute_intrinsic_common_line_angles
from common_lines.procrustes import nearest_rotation
from common_lines.radon import batch_common_line_sinograms

# Reuse the loading and RELION conversion used in the detection benchmark.
if __package__:
    from ._benchmark_preprocessing import (
        add_preprocessing_arguments,
        configuration_description,
        prepare_images,
        resolve_preprocessing_options,
    )
    from .detection_errors import read_images, read_metadata, symmetry_operators
else:
    from _benchmark_preprocessing import (
        add_preprocessing_arguments,
        configuration_description,
        prepare_images,
        resolve_preprocessing_options,
    )
    from detection_errors import read_images, read_metadata, symmetry_operators


EDGE_NAMES = ("ij", "ik", "jk")
CSV_COLUMNS = (
    "mode",
    "noise_std_deg",
    "particle_i",
    "particle_j",
    "particle_k",
    "status",
    "conditioning",
    "min_correlation",
    "mean_correlation",
    "handedness",
    "error_ij_deg",
    "error_ik_deg",
    "error_jk_deg",
    "mean_rotation_error_deg",
    "max_rotation_error_deg",
    "symmetry_operator_j",
    "symmetry_operator_k",
)


def ground_truth_angles(matrices):
    """Oriented common-line angles for unordered image pairs, in degrees.

    For each pair i<j the cross-product picks ONE shared 3D direction, so its
    two image-plane angles remain jointly oriented (both can be reversed).
    """
    n = len(matrices)
    first = np.full((n, n), np.nan)
    second = np.full((n, n), np.nan)
    i, j = np.triu_indices(n, k=1)
    cross = np.cross(matrices[i, 2], matrices[j, 2])
    valid = np.linalg.norm(cross, axis=1) > 1e-5
    if np.any(valid):
        x, y = compute_intrinsic_common_line_angles(
            matrices[i[valid]], matrices[j[valid]]
        )
        first[i[valid], j[valid]] = np.rad2deg(x) % 360
        second[i[valid], j[valid]] = np.rad2deg(y) % 360
    return first, second, valid.reshape(-1)


def perturb_angles(first, second, sigma, rng):
    """Independent angular noise at both ends of each undirected image pair."""
    n = len(first)
    i, j = np.triu_indices(n, k=1)
    a, b = first.copy(), second.copy()
    finite = np.isfinite(a[i, j]) & np.isfinite(b[i, j])
    a[i[finite], j[finite]] = (
        a[i[finite], j[finite]] + rng.normal(0, sigma, finite.sum())
    ) % 360
    b[i[finite], j[finite]] = (
        b[i[finite], j[finite]] + rng.normal(0, sigma, finite.sum())
    ) % 360
    return a, b


def detected_angles(
    matrices,
    names,
    embedded,
    metadata_path,
    image_stack,
    indices,
    theta,
    *,
    precenter="none",
    preprocessing_options=None,
    images=None,
):
    """Detect angles after optional pipeline-compatible image preparation."""
    if images is None:
        images = read_images(metadata_path, indices, names, embedded, image_stack)
    images = prepare_images(
        images, precenter=precenter, preprocessing_options=preprocessing_options
    )
    sinograms = batch_common_line_sinograms(images, theta=theta)
    graph = detect_common_lines_all_pairs(sinograms, allow_reversed=True)
    first, second = graph.oriented_angles(theta)
    return first, second, graph.correlation


def sample_triplets(matrices, maximum, rng):
    """Uniformly sample nondegenerate image triplets without replacement."""
    triplets = np.asarray(
        list(combinations(range(len(matrices)), 3)), dtype=int
    ).reshape(-1, 3)
    if len(triplets) == 0:
        raise ValueError("At least three images are required")
    i, j, k = triplets.T
    normals = matrices[:, 2, :]
    valid = (
        (np.linalg.norm(np.cross(normals[i], normals[j]), axis=1) > 1e-5)
        & (np.linalg.norm(np.cross(normals[i], normals[k]), axis=1) > 1e-5)
        & (np.linalg.norm(np.cross(normals[j], normals[k]), axis=1) > 1e-5)
    )
    triplets = triplets[valid]
    if len(triplets) == 0:
        raise ValueError("There are no triplets with three nondegenerate image pairs")
    if maximum and maximum < len(triplets):
        triplets = triplets[rng.choice(len(triplets), maximum, replace=False)]
    return np.sort(triplets, axis=1)


def triplet_measurements(i, j, k, first, second, correlation):
    """Angles as required by alignment.compute_transformation_matrices_from_angles."""
    angles = (
        first[i, j],
        second[i, j],
        first[i, k],
        second[i, k],
        first[j, k],
        second[j, k],
    )
    # Angular separation of the two lines *in each of the three images*.
    differences = (angles[2] - angles[0], angles[4] - angles[1], angles[5] - angles[3])
    conditioning = float(min(abs(np.sin(np.deg2rad(d))) for d in differences))
    corrs = (correlation[i, j], correlation[i, k], correlation[j, k])
    return angles, conditioning, float(np.min(corrs)), float(np.mean(corrs))


def rotation_error_deg(estimated, expected):
    cos_angle = (np.trace(estimated @ expected.T) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def symmetry_triplet_errors(estimates, matrices, symmetry_ops):
    """Best errors for a triplet under a *consistent* symmetry assignment.

    estimates are (Q_ij, Q_ik, Q_jk), mapping coordinates i->j, i->k,
    and j->k respectively. Orientations R_a map reference->image a.

    For a symmetry group G acting in reference coordinates, the physically
    equivalent orientations are R_a @ S_a. A common simultaneous right
    multiplication of all S_a does not change relative rotations, so set
    S_i=I and optimize over S_j, S_k. Hence the j->k truth MUST use
    S_k @ S_j.T, not an independently chosen group element.

    Returns (errors for the three edges, operator index for j, index for k).
    This is an oracle evaluation using ground truth, not part of estimation.
    """
    ri, rj, rk = matrices
    qij, qik, qjk = estimates
    ops = np.asarray(symmetry_ops)
    ej = rotation_errors_for_candidates(qij, rj @ ops @ ri.T)
    ek = rotation_errors_for_candidates(qik, rk @ ops @ ri.T)
    relative_symmetry = ops[:, None] @ np.swapaxes(ops[None, :], -1, -2)
    # Indexed [k,j]: R_k S_k S_j.T R_j.T.
    true_jk = rk @ relative_symmetry @ rj.T
    ejk = rotation_errors_for_candidates(qjk, true_jk)
    costs = ej[None, :] + ek[:, None] + ejk
    k_op, j_op = np.unravel_index(np.argmin(costs), costs.shape)
    return np.array([ej[j_op], ek[k_op], ejk[k_op, j_op]]), int(j_op), int(k_op)


def rotation_errors_for_candidates(estimate, truths):
    """Geodesic error, vectorized over symmetry candidate matrices."""
    trace = np.einsum("ab,...ab->...", estimate, truths)
    return np.rad2deg(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def evaluate_triplet(i, j, k, first, second, correlation, matrices, symmetry_ops=None):
    """Recover three relative rotations, with ONE oracle hand per triplet.

    Nearest-rotation projection is necessary for perturbed/inconsistent angles:
    the analytic angular-reconstitution solver can return nonorthogonal matrices.
    The main graph pipeline likewise projects averaged estimates onto SO(3).
    """
    if symmetry_ops is None:
        symmetry_ops = np.eye(3)[None, ...]
    angles, conditioning, min_corr, mean_corr = triplet_measurements(
        i, j, k, first, second, correlation
    )
    row = dict(
        status="ok",
        conditioning=conditioning,
        min_correlation=min_corr,
        mean_correlation=mean_corr,
        handedness="",
        error_ij_deg=np.nan,
        error_ik_deg=np.nan,
        error_jk_deg=np.nan,
        mean_rotation_error_deg=np.nan,
        max_rotation_error_deg=np.nan,
        symmetry_operator_j=-1,
        symmetry_operator_k=-1,
    )
    if not np.all(np.isfinite(angles)):
        row["status"] = "missing_angles"
        return row
    try:
        branches = compute_transformation_matrices_from_angles(*angles)
        branch_results = []
        for branch in branches:
            r_i, r_j, r_k = (nearest_rotation(r) for r in branch)
            estimates = (r_j @ r_i.T, r_k @ r_i.T, r_k @ r_j.T)
            branch_results.append(
                symmetry_triplet_errors(estimates, matrices[[i, j, k]], symmetry_ops)
            )
        # One oracle handedness AND mutually consistent symmetry assignments
        # per triplet. Never optimize symmetry/handedness edge by edge.
        hand = int(np.argmin([np.mean(item[0]) for item in branch_results]))
        errors, j_op, k_op = branch_results[hand]
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        row["status"] = "reconstruction_failed"
        return row

    row.update(
        handedness=str(hand),
        error_ij_deg=errors[0],
        error_ik_deg=errors[1],
        error_jk_deg=errors[2],
        mean_rotation_error_deg=errors.mean(),
        max_rotation_error_deg=errors.max(),
        symmetry_operator_j=j_op,
        symmetry_operator_k=k_op,
    )
    return row


def evaluate_mode(
    label,
    sigma,
    triplets,
    first,
    second,
    correlation,
    matrices,
    indices,
    symmetry_ops=None,
):
    rows = []
    for i, j, k in triplets:
        row = {
            "mode": label,
            "noise_std_deg": sigma,
            "particle_i": int(indices[i] + 1),
            "particle_j": int(indices[j] + 1),
            "particle_k": int(indices[k] + 1),
        }
        row.update(
            evaluate_triplet(
                i, j, k, first, second, correlation, matrices, symmetry_ops=symmetry_ops
            )
        )
        rows.append(row)
    return rows


def print_summary(rows, label):
    errors = np.array([r["mean_rotation_error_deg"] for r in rows], dtype=float)
    good = np.isfinite(errors)
    failed = len(rows) - int(good.sum())
    print(f"\n{label}:")
    print(f"  Triplets attempted:           {len(rows)}")
    print(f"  Successful reconstructions:   {good.sum()} ({100 * good.mean():.1f}%)")
    print(f"  Failed reconstructions:       {failed}")
    if not good.any():
        return
    values = errors[good]
    print(f"  Mean triplet rotation error:  {values.mean():.2f} deg")
    print(f"  Median triplet error:         {np.median(values):.2f} deg")
    print(f"  Triplets within 10 deg:       {100 * np.mean(values <= 10):.1f}%")
    print(f"  Triplets within 20 deg:       {100 * np.mean(values <= 20):.1f}%")
    print(
        f"  All 3 edges within 20 deg:    {100 * np.mean(np.array([r['max_rotation_error_deg'] for r in rows])[good] <= 20):.1f}%"
    )


def save_results(prefix, grouped):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [r for rows in grouped.values() for r in rows]
    import csv

    with prefix.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(0, 180, 37)
    for label, rows in grouped.items():
        good = [r for r in rows if np.isfinite(r["mean_rotation_error_deg"])]
        if not good:
            continue
        errors = np.array([r["mean_rotation_error_deg"] for r in good])
        axes[0].hist(
            errors,
            bins=bins,
            histtype="step",
            linewidth=1.7,
            weights=np.full(len(errors), 1.0 / len(errors)),
            label=label,
        )
        # Conditioning comes from the actual (possibly perturbed) input angles.
        cond = np.array([r["conditioning"] for r in good])
        axes[1].scatter(cond, errors, s=9, alpha=0.28, label=label)
    axes[0].set(
        title="Rotation-error distribution",
        xlabel="Mean rotation error per triplet (deg)",
        ylabel="Fraction of successful triplets per bin",
    )
    axes[1].set(
        title="Triplet conditioning versus error",
        xlabel="Min. |sin(angle separation)|",
        ylabel="Mean rotation error (deg)",
        xlim=(-0.02, 1.02),
    )
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"\nSaved: {prefix.with_suffix('.csv')} and {prefix.with_suffix('.png')}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "metadata", type=Path, help="STAR or NPZ with ground-truth orientations"
    )
    parser.add_argument(
        "--image-stack", type=Path, help="Optional MRC/MRCS images in metadata order"
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=24,
        help="Sample at most this many images (default: 24)",
    )
    parser.add_argument(
        "--max-triplets",
        type=int,
        default=1000,
        help="Maximum random triplets (default: 1000; 0 means all)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("exact", "perturbed", "detected"),
        default=("exact", "perturbed", "detected"),
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 5.0, 10.0),
        help="Gaussian angular-noise standard deviations in degrees (default: 1 2 5 10)",
    )
    parser.add_argument(
        "--angle-step",
        type=float,
        default=2.0,
        help="Detector angular step, in degrees",
    )
    parser.add_argument(
        "--symmetry",
        default="C1",
        help="Reference-object symmetry, Cn or Dn (default: C1)",
    )
    add_preprocessing_arguments(parser)
    parser.add_argument(
        "--pipeline-preprocessing",
        action="store_true",
        help="Legacy shorthand for --precenter center_of_mass "
        "--preprocessing default (not the current pipeline defaults)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Image/triplet/noise sampling seed"
    )
    parser.add_argument("--output", type=Path, default=Path("orientation_validation"))
    args = parser.parse_args()
    operators = symmetry_operators(args.symmetry)

    if args.pipeline_preprocessing:
        if (
            args.precenter != "none"
            or args.preprocessing != "none"
            or any(
                v is not None
                for v in (args.mask, args.subtract_background, args.clip_negative)
            )
            or args.mask_radius_fraction != 1.0
        ):
            parser.error(
                "--pipeline-preprocessing cannot be combined with the new "
                "preprocessing flags; use --precenter and --preprocessing instead"
            )
        args.precenter = "center_of_mass"
        args.preprocessing = "default"

    if args.n_images < 3 or args.max_triplets < 0 or args.angle_step <= 0:
        parser.error("--n-images must be >= 3; --max-triplets >= 0; --angle-step > 0")
    if any((not np.isfinite(x)) or x < 0 for x in args.noise_std):
        parser.error("--noise-std values must be finite and nonnegative")
    matrices, names, embedded = read_metadata(args.metadata)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("Expected rotation matrices with shape (N,3,3)")
    if len(matrices) < 3:
        raise ValueError("At least three orientations are required")
    rng = np.random.default_rng(args.seed)
    indices = np.sort(
        rng.choice(len(matrices), size=min(args.n_images, len(matrices)), replace=False)
    )
    selected_matrices = matrices[indices]
    triplets = sample_triplets(selected_matrices, args.max_triplets, rng)
    first, second, _ = ground_truth_angles(selected_matrices)
    dummy_corr = np.full_like(first, np.nan)
    grouped = {}

    print(f"Symmetry: {args.symmetry.upper()} (consistent triplet scoring)")
    print(f"Images selected: {len(indices)}; triplets selected: {len(triplets)}")
    if "exact" in args.modes:
        label = "exact"
        rows = evaluate_mode(
            label,
            0.0,
            triplets,
            first,
            second,
            dummy_corr,
            selected_matrices,
            indices,
            operators,
        )
        grouped[label] = rows
        print_summary(rows, label)
    if "perturbed" in args.modes:
        for sigma in args.noise_std:
            label = f"noise_{sigma:g}deg"
            a, b = perturb_angles(first, second, sigma, rng)
            rows = evaluate_mode(
                label,
                sigma,
                triplets,
                a,
                b,
                dummy_corr,
                selected_matrices,
                indices,
                operators,
            )
            grouped[label] = rows
            print_summary(rows, label)
    if "detected" in args.modes:
        theta = np.arange(0.0, 180.0, args.angle_step)
        # Resolve against the same selected image dimensions as the detector.
        images = read_images(args.metadata, indices, names, embedded, args.image_stack)
        options = resolve_preprocessing_options(images, args)
        print(f"Image preparation: {configuration_description(args)}")
        a, b, corr = detected_angles(
            matrices,
            names,
            embedded,
            args.metadata,
            args.image_stack,
            indices,
            theta,
            precenter=args.precenter,
            preprocessing_options=options,
            images=images,
        )
        label = "detected"
        rows = evaluate_mode(
            label, np.nan, triplets, a, b, corr, selected_matrices, indices, operators
        )
        grouped[label] = rows
        print_summary(rows, label)

    save_results(args.output, grouped)


if __name__ == "__main__":
    main()
