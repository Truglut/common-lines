"""
Compare common lines detected from images against known orientations.

Example:
    python detection_errors.py particles.npz --n-images 24
    python detection_errors.py particles.star --n-images 24 --symmetry D7
    python detection_errors.py particles.star --image-stack images.mrcs --symmetry C5
    python detection_errors.py particles.npz --symmetry D7 --uniform-baseline --permutation-test
    python -m scripts.detection_errors particles.npz --precenter center_of_mass --preprocessing default
    python -m scripts.detection_errors particles.npz --preprocessing default --no-mask --clip-negative

Optional baselines use 200 Monte Carlo trials (BASELINE_TRIALS below) and the
existing symmetry-aware pair-error definition. --seed makes runs reproducible.

Requires numpy, matplotlib, mrcfile and the common_lines package; STAR
files additionally require starfile.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np

from common_lines.detection import detect_common_lines_all_pairs
from common_lines.radon import batch_common_line_sinograms
from common_lines.forward import compute_intrinsic_common_line_angles

if __package__:
    from ._benchmark_preprocessing import (
        add_preprocessing_arguments,
        configuration_description,
        prepare_images,
        resolve_preprocessing_options,
    )
else:
    from _benchmark_preprocessing import (
        add_preprocessing_arguments,
        configuration_description,
        prepare_images,
        resolve_preprocessing_options,
    )


# Increase this constant if tighter Monte Carlo estimates are needed.
BASELINE_TRIALS = 200
HIST_BINS = np.linspace(0.0, 180.0, 37)


def euler_zyz_to_matrix(
    rot: np.ndarray, tilt: np.ndarray, psi: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """
    Convert ZYZ Euler angles to rotation matrices. Input angles are given in
    radians and may have any broadcast-compatible shapes.

    Parameters
    ----------
    rot : np.ndarray
        First rotation angle about the z axis, in radians.
    tilt : np.ndarray
        Rotation angle about the y axis, in radians.
    psi : np.ndarray
        Final rotation angle about the z axis, in radians.
    out : np.ndarray, optional
        Array in which to store the resulting matrices. It must have
        shape ``broadcast_shape + (3, 3)`` and the same dtype as ``rot``.

    Returns
    -------
    np.ndarray
        Rotation matrices with shape ``broadcast_shape + (3, 3)``.

    Notes
    -----
    With the convention used by this implementation, the projection direction is
    given by the third row of the rotation matrix. This can be extracted by
    doing ``euler_zyz_to_matrix(...)[..., 2, :]``.
    """
    # Create the output
    batch_shape = np.broadcast_shapes(rot.shape, tilt.shape, psi.shape)
    result_shape = batch_shape + (3, 3)
    dtype = rot.dtype

    if out is None:
        out = np.empty(result_shape, dtype=dtype)
    elif out.shape != result_shape or out.dtype != dtype:
        raise RuntimeError("Invalid output array was provided")

    ai = rot
    aj = tilt
    ak = psi

    # Obtain sin and cos of the angles
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)

    # Obtain the combinations
    cc = cj * ci
    cs = cj * si
    sc = sj * ci
    ss = sj * si

    # Build the matrix
    out[..., 0, 0] = ck * cc - sk * si
    out[..., 0, 1] = ck * cs + sk * ci
    out[..., 0, 2] = -ck * sj
    out[..., 1, 0] = -sk * cc - ck * si
    out[..., 1, 1] = -sk * cs + ck * ci
    out[..., 1, 2] = sk * sj
    out[..., 2, 0] = sc
    out[..., 2, 1] = ss
    out[..., 2, 2] = cj

    return out


def star_angles_to_matrices(particles):
    """Return (N, 3, 3) matrices with v_image = R @ v_reference.

    IMPORTANT: Implement this using your project's RELION Euler convention.
    In particular, verify the Rot/Tilt/Psi order, active/passive convention,
    and image y-axis direction against a known orientation before running a
    quantitative validation. A guessed conversion would make results misleading.
    """
    rot = np.deg2rad(particles["angleRot"].to_numpy())
    tilt = np.deg2rad(particles["angleTilt"].to_numpy())
    psi = np.deg2rad(particles["anglePsi"].to_numpy())
    return euler_zyz_to_matrix(rot, tilt, psi)


def read_metadata(path):
    """Return (rotation_matrices, image_names, embedded_images).

    .npz: expects 'matrices', optionally 'original_images' and 'image_names'.
    .star: expects RELION particle columns; image paths are 'index@path'.
    """
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            matrices = np.asarray(data["matrices"])
            names = (
                np.asarray(data["image_names"]).astype(str)
                if "image_names" in data
                else None
            )
            images = (
                np.asarray(data["original_images"])
                if "original_images" in data
                else None
            )
        return matrices, names, images

    if path.suffix.lower() == ".star":
        import starfile

        data = starfile.read(path)
        particles = data["particles"] if isinstance(data, dict) else data
        names = particles["image"].to_numpy(dtype=str) if "image" in particles else None
        return star_angles_to_matrices(particles), names, None

    raise ValueError("Metadata must be a .star or .npz file")


def read_images(metadata_path, indices, image_names, embedded_images, image_stack):
    """Load only selected images; metadata and image stack share row order."""
    if image_stack is None and embedded_images is not None:
        return np.asarray(embedded_images[indices], dtype=np.float32)

    if image_stack is not None:
        # When explicitly supplying one stack, row k refers to stack slice k.
        with mrcfile.open(image_stack, permissive=True) as mrc:
            return np.asarray(mrc.data[indices], dtype=np.float32).copy()

    if image_names is None:
        raise ValueError("No image data supplied and metadata has no image names")

    images = []
    for name in image_names[indices]:
        # RELION-style 1-based slice index, e.g. 000023@Particles/stack.mrcs
        index_text, path_text = str(name).split("@", 1)
        stack_path = Path(path_text)
        if not stack_path.is_absolute():
            relative_to_metadata = metadata_path.parent / stack_path
            stack_path = (
                relative_to_metadata if relative_to_metadata.exists() else stack_path
            )
        with mrcfile.open(stack_path, permissive=True) as mrc:
            images.append(np.array(mrc.data[int(index_text) - 1], dtype=np.float32))
    return np.stack(images)


def symmetry_operators(symmetry):
    """RELION Cn/Dn operators in reference coordinates (identity first).

    Cn: rotations by 2*pi*k/n about Z.
    Dn: those rotations, plus each followed by a 180-degree rotation about X.
    """
    match = re.fullmatch(r"([CD])([1-9][0-9]*)", symmetry.upper())
    if match is None:
        raise ValueError("--symmetry must be Cn or Dn, e.g. C1, C7 or D7")
    group, n_text = match.groups()
    n = int(n_text)

    angles = 2.0 * np.pi * np.arange(n) / n
    c, s = np.cos(angles), np.sin(angles)
    cyclic = np.zeros((n, 3, 3))
    cyclic[:, 0, 0] = c
    cyclic[:, 0, 1] = -s
    cyclic[:, 1, 0] = s
    cyclic[:, 1, 1] = c
    cyclic[:, 2, 2] = 1.0

    if group == "C":
        return cyclic

    twofold_x = np.diag([1.0, -1.0, -1.0])
    return np.concatenate((cyclic, cyclic @ twofold_x), axis=0)


def wrap_degrees(angle):
    """Signed angular difference in [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def compare_angles(truth, measured):
    """Signed angle errors, allowing only a simultaneous 180-degree reversal.

    Arrays may have broadcast-compatible leading dimensions; the last dimension
    contains the two angles of a common-line pair.
    """
    forward = wrap_degrees(measured - truth)
    backward = wrap_degrees(measured - (truth + 180.0))
    use_backward = np.abs(backward).mean(axis=-1) < np.abs(forward).mean(axis=-1)
    return np.where(use_backward[..., None], backward, forward)


def candidate_common_line_angles(matrices, i, j, symmetry_ops):
    """Return the (pair, symmetry, two angles) ground truths and valid mask."""
    truth = np.empty((len(i), len(symmetry_ops), 2))
    valid = np.empty((len(i), len(symmetry_ops)), dtype=bool)

    for k, operator in enumerate(symmetry_ops):
        equivalent_j = matrices[j] @ operator
        cross = np.cross(matrices[i, 2, :], equivalent_j[:, 2, :])
        valid[:, k] = np.linalg.norm(cross, axis=-1) > 1e-5
        first, second = compute_intrinsic_common_line_angles(matrices[i], equivalent_j)
        truth[:, k] = np.rad2deg(np.column_stack((first, second)))

    return truth, valid


def best_candidate_errors(candidate_truth, valid, measured):
    """Choose the symmetry equivalent minimizing mean absolute pair-angle error.

    Pairs with no valid geometric common line receive NaN errors. This matters
    for permutation trials when some reassigned orientations become coincident.
    """
    candidate_errors = compare_angles(candidate_truth, measured[:, None, :])
    scores = np.abs(candidate_errors).mean(axis=-1)
    scores[~valid] = np.inf
    selected = scores.argmin(axis=1)
    errors = candidate_errors[np.arange(len(selected)), selected].copy()
    errors[~valid.any(axis=1)] = np.nan
    return errors, selected


def validate(matrices, images, theta, symmetry_ops=None):
    """Compare detected lines against all symmetry-equivalent ground truths.

    With v_image = R @ v_reference, an equivalent orientation is R_j @ S.
    Expanding only image j suffices to enumerate relative symmetry equivalents.
    """
    if symmetry_ops is None:
        symmetry_ops = np.eye(3)[None, ...]

    i, j = np.triu_indices(len(matrices), k=1)
    candidate_truth, valid = candidate_common_line_angles(matrices, i, j, symmetry_ops)

    # Coincident planes have no uniquely defined common-line direction.
    usable = valid.any(axis=1)
    i, j = i[usable], j[usable]
    candidate_truth, valid = candidate_truth[usable], valid[usable]
    if len(i) == 0:
        raise ValueError("No nondegenerate image pairs were selected")

    sinograms = batch_common_line_sinograms(images, theta=theta)
    graph = detect_common_lines_all_pairs(sinograms, allow_reversed=True)
    measured_i, measured_j = graph.oriented_angles(theta)
    measured = np.column_stack((measured_i[i, j], measured_j[i, j]))

    errors, selected = best_candidate_errors(candidate_truth, valid, measured)
    truth = candidate_truth[np.arange(len(i)), selected]
    return i, j, truth, measured, errors, graph.correlation[i, j], selected


def pair_errors(errors):
    """Mean absolute error of each valid angle pair, in degrees."""
    return np.abs(errors[np.isfinite(errors).all(axis=-1)]).mean(axis=-1)


def summary_metrics(errors):
    """Mean angle error, median pair error, and fraction with pair error <= 10 deg."""
    per_pair = pair_errors(errors)
    if len(per_pair) == 0:
        return np.full(3, np.nan)
    return np.array([per_pair.mean(), np.median(per_pair), np.mean(per_pair <= 10.0)])


def run_uniform_baseline(candidate_truth, valid, theta, rng):
    """Random independent angles drawn from the detector's sampled angle grid.

    The first angle is in [0, 180), as in the detector. The second is drawn
    from that grid with an independent 180-degree reversal, as permitted by
    allow_reversed=True. No new images or correlations are generated.
    """
    n_pairs = len(candidate_truth)
    metrics = np.empty((BASELINE_TRIALS, 3))
    histogram = np.zeros(len(HIST_BINS) - 1)

    for trial in range(BASELINE_TRIALS):
        random_angles = rng.choice(theta, size=(n_pairs, 2)).astype(float)
        random_angles[:, 1] += 180.0 * rng.integers(0, 2, size=n_pairs)
        errors, _ = best_candidate_errors(candidate_truth, valid, random_angles)
        per_pair = pair_errors(errors)
        metrics[trial] = summary_metrics(errors)
        histogram += np.histogram(per_pair, HIST_BINS)[0] / len(per_pair)

    return {"metrics": metrics, "histogram": histogram / BASELINE_TRIALS}


def run_permutation_baseline(matrices, i, j, measured, symmetry_ops, rng):
    """Shuffle complete orientation assignments among image nodes.

    Detected angle pairs, image-pair edges, and correlations remain unchanged.
    This destroys the image-to-orientation correspondence, not the graph.
    """
    metrics = np.empty((BASELINE_TRIALS, 3))
    histogram = np.zeros(len(HIST_BINS) - 1)
    valid_counts = np.empty(BASELINE_TRIALS, dtype=int)

    for trial in range(BASELINE_TRIALS):
        permuted = matrices[rng.permutation(len(matrices))]
        truth, valid = candidate_common_line_angles(permuted, i, j, symmetry_ops)
        errors, _ = best_candidate_errors(truth, valid, measured)
        per_pair = pair_errors(errors)
        valid_counts[trial] = len(per_pair)
        metrics[trial] = summary_metrics(errors)
        if len(per_pair):
            histogram += np.histogram(per_pair, HIST_BINS)[0] / len(per_pair)

    n_nonempty = np.count_nonzero(valid_counts)
    return {
        "metrics": metrics,
        "histogram": histogram / n_nonempty if n_nonempty else histogram,
        "valid_counts": valid_counts,
        "total_pairs": len(i),
    }


def _ranks(values):
    """Average ranks, including ties (without requiring scipy)."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    return ((starts + ends + 1) / 2.0)[inverse]


def _pearson(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def correlation_diagnostics(correlation, per_pair):
    """Linear association, monotonic association, and top/bottom quartile hit rates."""
    good = np.isfinite(correlation) & np.isfinite(per_pair)
    corr, errors = correlation[good], per_pair[good]
    pearson = _pearson(corr, errors)
    spearman = _pearson(_ranks(corr), _ranks(errors))

    low_rate = high_rate = np.nan
    if len(corr) >= 4:
        q1, q3 = np.quantile(corr, [0.25, 0.75])
        if q1 < q3:
            low_rate = float(np.mean(errors[corr <= q1] <= 10.0))
            high_rate = float(np.mean(errors[corr >= q3] <= 10.0))
    return pearson, spearman, low_rate, high_rate


def _print_baseline(name, baseline, observed, *, permutation=False):
    trial_metrics = baseline["metrics"]
    good = np.isfinite(trial_metrics[:, 0])
    if not np.any(good):
        print(f"\n{name}: no usable baseline trials")
        return
    trial_metrics = trial_metrics[good]
    means = trial_metrics.mean(axis=0)
    print(f"\n{name} ({len(trial_metrics)} trials):")
    print(
        f"  Mean absolute angle error: {means[0]:.2f} deg "
        f"(trial SD {trial_metrics[:, 0].std():.2f} deg)"
    )
    print(f"  Median pair error:         {means[1]:.2f} deg")
    print(f"  Pairs within 10 deg:       {100 * means[2]:.1f}%")

    if permutation:
        # Inclusive, finite-sample Monte Carlo p-values (smaller mean is better;
        # larger within-10 fraction is better).
        p_mean = (1 + np.sum(trial_metrics[:, 0] <= observed[0])) / (
            1 + len(trial_metrics)
        )
        p_within = (1 + np.sum(trial_metrics[:, 2] >= observed[2])) / (
            1 + len(trial_metrics)
        )
        print(f"  Permutation p (mean error):     {p_mean:.4f}")
        print(f"  Permutation p (within 10 deg):  {p_within:.4f}")
        counts = baseline["valid_counts"]
        if np.any(counts < baseline["total_pairs"]):
            print(
                f"  Valid pairs per trial: {counts.min()}–{counts.max()} "
                "(some reassigned pairs are degenerate)"
            )
    else:
        normalized = 1 - observed[0] / means[0] if means[0] > 0 else np.nan
        adjusted = (observed[2] - means[2]) / (1 - means[2]) if means[2] < 1 else np.nan
        print(f"  Chance-normalized mean error:   {normalized:.3f}")
        print(f"  Chance-adjusted within 10 deg:  {adjusted:.3f}")


def save_results(
    prefix,
    original_indices,
    i,
    j,
    truth,
    measured,
    errors,
    correlation,
    selected_ops,
    symmetry,
    uniform=None,
    permutation=None,
):
    per_pair = pair_errors(errors)
    observed = summary_metrics(errors)
    rows = np.column_stack(
        (
            original_indices[i] + 1,
            original_indices[j] + 1,
            truth,
            measured,
            errors,
            per_pair,
            correlation,
            selected_ops,
        )
    )
    columns = (
        "particle_i,particle_j,truth_i_deg,truth_j_deg,"
        "detected_i_deg,detected_j_deg,error_i_deg,error_j_deg,"
        "pair_mean_abs_error_deg,correlation,symmetry_operator_index"
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        prefix.with_suffix(".csv"),
        rows,
        delimiter=",",
        header=columns,
        comments="",
        fmt=["%d", "%d"] + ["%.5f"] * 8 + ["%d"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Common-line validation ({symmetry.upper()})")
    axes[0].hist(
        per_pair,
        bins=HIST_BINS,
        weights=np.full(len(per_pair), 1 / len(per_pair)),
        alpha=0.55,
        label="Detected",
    )
    if uniform is not None:
        axes[0].stairs(
            uniform["histogram"], HIST_BINS, label="Uniform random", linewidth=1.8
        )
    if permutation is not None:
        axes[0].stairs(
            permutation["histogram"],
            HIST_BINS,
            label="Permuted orientations",
            linewidth=1.8,
        )
    if uniform is not None or permutation is not None:
        axes[0].legend()
    axes[0].set(
        xlabel="Mean absolute angle error per pair (deg)",
        ylabel="Fraction of pairs per bin",
        title="Angular-error distribution",
    )
    axes[1].scatter(correlation, per_pair, s=12, alpha=0.6)
    axes[1].set(
        xlabel="Detected correlation",
        ylabel="Mean absolute error (deg)",
        title="Error versus match correlation",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(prefix.with_suffix(".png"), dpi=160)
    plt.close(fig)

    if permutation is not None:
        # A histogram of *trial mean errors* is distinct from the pair-error
        # histogram: it visualizes the null distribution of the overall score.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            permutation["metrics"][:, 0][np.isfinite(permutation["metrics"][:, 0])],
            bins=25,
            label="Permutations",
        )
        ax.axvline(observed[0], color="tab:red", linewidth=2, label="Detected")
        ax.set(
            xlabel="Mean absolute angle error (deg)",
            ylabel="Number of permutations",
            title="Permutation-test null distribution",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(prefix.with_name(prefix.name + "_permutation.png"), dpi=160)
        plt.close(fig)

    print(f"Symmetry: {symmetry.upper()}")
    print(f"Pairs evaluated: {len(per_pair)}")
    print(f"Mean absolute angle error: {observed[0]:.2f} deg")
    print(f"Median pair error:         {observed[1]:.2f} deg")
    print(f"Pairs within 10 deg:       {100 * observed[2]:.1f}%")

    pearson, spearman, low_rate, high_rate = correlation_diagnostics(
        correlation, per_pair
    )
    print("\nDetection correlation vs pair error (negative is better):")
    print(f"  Pearson r:               {pearson:.3f}")
    print(f"  Spearman rho:            {spearman:.3f}")
    print(
        f"  Within 10 deg, lowest/highest correlation quartiles: "
        f"{100 * low_rate:.1f}% / {100 * high_rate:.1f}%"
    )

    if uniform is not None:
        _print_baseline("Uniform-angle baseline", uniform, observed)
    if permutation is not None:
        _print_baseline(
            "Orientation-permutation baseline", permutation, observed, permutation=True
        )

    print(f"\nSaved: {prefix.with_suffix('.csv')} and {prefix.with_suffix('.png')}")
    if permutation is not None:
        print(f"Saved: {prefix.with_name(prefix.name + '_permutation.png')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="STAR or NPZ orientation metadata")
    parser.add_argument(
        "--image-stack", type=Path, help="Optional MRC/MRCS stack in metadata order"
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=24,
        help="Randomly sample this many images (default 24)",
    )
    parser.add_argument(
        "--angle-step",
        type=float,
        default=2.0,
        help="Sinogram angle spacing in degrees",
    )
    parser.add_argument(
        "--symmetry",
        default="C1",
        help="RELION symmetry Cn or Dn, e.g. C1, C7, D7 (default C1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("common_line_validation"))
    add_preprocessing_arguments(parser)
    parser.add_argument(
        "--uniform-baseline",
        action="store_true",
        help="Compare with random uniform detected angles and report chance-adjusted scores",
    )
    parser.add_argument(
        "--permutation-test",
        action="store_true",
        help="Shuffle image-to-orientation assignments, report p-values and plot the null distribution",
    )
    args = parser.parse_args()
    operators = symmetry_operators(args.symmetry)

    matrices, names, embedded = read_metadata(args.metadata)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("Expected matrices with shape (N, 3, 3)")
    if args.n_images < 2 or args.angle_step <= 0:
        raise ValueError("--n-images must be >= 2 and --angle-step must be positive")

    rng = np.random.default_rng(args.seed)
    indices = np.sort(
        rng.choice(len(matrices), size=min(len(matrices), args.n_images), replace=False)
    )
    images = read_images(args.metadata, indices, names, embedded, args.image_stack)
    matrices = matrices[indices]

    options = resolve_preprocessing_options(images, args)
    images = prepare_images(
        images, precenter=args.precenter, preprocessing_options=options
    )
    print(f"Image preparation: {configuration_description(args)}")
    theta = np.arange(0.0, 180.0, args.angle_step)
    i, j, truth, measured, errors, correlation, selected_ops = validate(
        matrices, images, theta, operators
    )
    uniform = None
    if args.uniform_baseline:
        candidate_truth, valid = candidate_common_line_angles(matrices, i, j, operators)
        uniform = run_uniform_baseline(
            candidate_truth, valid, theta, np.random.default_rng(args.seed + 1)
        )
    permutation = (
        run_permutation_baseline(
            matrices, i, j, measured, operators, np.random.default_rng(args.seed + 2)
        )
        if args.permutation_test
        else None
    )
    save_results(
        args.output,
        indices,
        i,
        j,
        truth,
        measured,
        errors,
        correlation,
        selected_ops,
        args.symmetry,
        uniform,
        permutation,
    )


if __name__ == "__main__":
    main()
