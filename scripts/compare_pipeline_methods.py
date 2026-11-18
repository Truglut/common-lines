"""Compare all six reconstitution and handedness method combinations.

From the repository root:
    python -m scripts.compare_pipeline_methods \
        --datasets-dir data/class-averages/converted --jobs 4

Uses every image, every valid third-image witness, the stored class averages,
a 2-degree angular grid, and seed 0 by default. Each run invokes
scripts.pipeline_errors, which evaluates the public pipeline's relative
rotations using the dataset's point group. The dataset directory must contain
a datasets.csv with a file and symmetry entry for every NPZ stack.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_dataset_symmetries(path: Path, files: list[Path]) -> dict[str, str]:
    """Read one point group per NPZ filename."""
    filenames = {file.name for file in files}
    symmetries: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source, strict=True)
        if not {"file", "symmetry"}.issubset(rows.fieldnames or ()):
            raise ValueError(f"{path} must include file and symmetry columns")
        for row in rows:
            filename = (row["file"] or "").strip()
            symmetry = (row["symmetry"] or "").strip().upper()
            if filename not in filenames:
                raise ValueError(
                    f"{path}:{rows.line_num}: unknown NPZ file {filename!r}"
                )
            if filename in symmetries:
                raise ValueError(
                    f"{path}:{rows.line_num}: duplicate entry for {filename}"
                )
            if not symmetry:
                raise ValueError(
                    f"{path}:{rows.line_num}: missing symmetry for {filename}"
                )
            symmetries[filename] = symmetry
    missing = filenames - symmetries.keys()
    if missing:
        raise ValueError(f"{path} has no entries for {', '.join(sorted(missing))}")
    return symmetries


def run_one(command: list[str], prefix: Path, environment: dict[str, str]) -> dict:
    """Run one CLI and summarize all requested directed edges, including failures."""
    started = time.monotonic()
    with prefix.with_suffix(".log").open("w") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    elapsed = time.monotonic() - started
    with prefix.with_suffix(".csv").open() as file:
        edges = list(csv.DictReader(file))
    errors = np.array([float(row["rotation_error_deg"]) for row in edges])
    valid = np.array([row["valid_rotation"] == "1" for row in edges])
    finite = valid & np.isfinite(errors)
    fallback = np.array([row["used_fallback"] == "1" for row in edges])
    inverse = np.array([float(row["inverse_disagreement_deg"]) for row in edges])
    with np.load(prefix.with_suffix(".npz")) as saved:
        configuration = json.loads(str(saved["run_configuration_json"]))
        n_images = len(saved["selected_indices"])
        mirrored = bool(saved["oracle_mirrored_for_scoring"])
    result = {
        "dataset": Path(configuration["metadata"]).stem,
        "n_images": n_images,
        "symmetry": configuration["symmetry"],
        "max_third": configuration["max_third"],
        "reconstitution_method": configuration["reconstitution_method"],
        "handedness_method": configuration["handedness_method"],
        "seed": configuration["seed"],
        "edges": len(errors),
        "invalid": int((~finite).sum()),
        "fallback": int(fallback.sum()),
        "mean_deg": float(errors[finite].mean()),
        "median_deg": float(np.median(errors[finite])),
        "p90_deg": float(np.percentile(errors[finite], 90)),
        "inverse_mean_deg": float(np.nanmean(inverse)),
        "oracle_mirrored_for_scoring": mirrored,
        "seconds": elapsed,
        "prefix": str(prefix.relative_to(ROOT))
        if prefix.is_relative_to(ROOT)
        else str(prefix),
    }
    for threshold in (5, 10, 20, 30):
        result[f"within{threshold}_pct"] = float(
            100 * np.count_nonzero(finite & (errors <= threshold)) / len(errors)
        )
    for label, subset in (("nonfallback", ~fallback), ("fallback", fallback)):
        result[f"{label}_within20_pct"] = (
            float(
                100 * np.count_nonzero(subset & finite & (errors <= 20)) / subset.sum()
            )
            if subset.any()
            else None
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        required=True,
        help="Directory containing NPZ stacks and their datasets.csv metadata",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/j_sync_comparison"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("--jobs must be positive and --seeds must be distinct")
    datasets_dir = args.datasets_dir.resolve()
    files = sorted(datasets_dir.glob("*.npz"))
    if not files:
        parser.error("No NPZ datasets found")
    datasets_csv = datasets_dir / "datasets.csv"
    try:
        symmetries = read_dataset_symmetries(datasets_csv, files)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        NUMBA_NUM_THREADS="2",
        MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / "common-lines-mpl"),
        MPLBACKEND="Agg",
    )
    runs: list[tuple[list[str], Path]] = []
    for dataset in files:
        with np.load(dataset) as data:
            n_images = len(data["matrices"])
        if n_images < 3:
            parser.error(f"At least three images required: {dataset}")
        symmetry = symmetries[dataset.name]
        max_third = n_images - 2
        for seed in args.seeds:
            for method in ("averaging", "voting", "soft_voting"):
                for handedness in ("star", "synchronization"):
                    prefix = (
                        output
                        / "pipeline"
                        / f"{dataset.stem}_full_seed{seed}_{method}_{handedness}"
                    )
                    prefix.parent.mkdir(parents=True, exist_ok=True)
                    command = [
                        sys.executable,
                        "-m",
                        "scripts.pipeline_errors",
                        str(dataset),
                        "--n-images",
                        str(n_images),
                        "--max-third",
                        str(max_third),
                        "--angle-step",
                        "2",
                        "--symmetry",
                        symmetry,
                        "--seed",
                        str(seed),
                        "--precenter",
                        "none",
                        "--preprocessing",
                        "none",
                        "--reconstitution-method",
                        method,
                        "--handedness-method",
                        handedness,
                        "--output",
                        str(prefix),
                    ]
                    runs.append((command, prefix))

    sources = sorted((ROOT / "src/common_lines").rglob("*.py")) + sorted(
        (ROOT / "scripts").glob("*.py")
    )
    manifest = {
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "numba", "matplotlib")
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [datasets_csv, *files, *sources]
        },
        "commands": [command for command, _ in runs],
        "jobs": args.jobs,
        "environment": {
            k: environment[k]
            for k in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "MPLCONFIGDIR",
                "MPLBACKEND",
            )
        },
        "timing_note": "End-to-end subprocess wall time including startup and output; concurrent jobs may compete for resources.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "source.patch").write_text(
        subprocess.check_output(["git", "diff"], cwd=ROOT, text=True)
    )
    results: list[dict] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_one, command, prefix, environment): prefix
            for command, prefix in runs
        }
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            results.sort(
                key=lambda r: (
                    r["dataset"],
                    r["seed"],
                    r["reconstitution_method"],
                    r["handedness_method"],
                )
            )
            with (output / "summary.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
            print(
                f"[{len(results)}/{len(runs)}] {futures[future].name}: {row['within20_pct']:.2f}% within 20°, {row['seconds']:.1f}s",
                flush=True,
            )
    manifest["total_seconds"] = time.monotonic() - started
    manifest["completed_runs"] = len(results)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
