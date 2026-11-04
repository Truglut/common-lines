"""Run all eight aggregation/handedness/witness-budget configurations.

From the repository root:
    python -m scripts.compare_pipeline_methods --jobs 4

Uses every image, raw-image preparation, a 2-degree angular grid, and seeds
0/1/2 by default. Proteasome uses D7 evaluation symmetry, TRPV1 C4, and other
datasets C1, matching the earlier manually-selected-data comparison. Each run
invokes scripts.pipeline_errors, including the complete public pipeline.
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


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
        "budget": "half" if configuration["n_third"] == n_images // 2 else "full",
        "n_third": configuration["n_third"],
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
        default=ROOT / "data/daniel/classes/manually-selected",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/j_sync_comparison"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("--jobs must be positive and --seeds must be distinct")
    files = sorted(args.datasets_dir.resolve().glob("*.npz"))
    if not files:
        parser.error("No NPZ datasets found")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        NUMBA_NUM_THREADS="2",
        MPLCONFIGDIR="/tmp/common-lines-mpl",
        MPLBACKEND="Agg",
    )
    runs: list[tuple[list[str], Path]] = []
    for dataset in files:
        with np.load(dataset) as data:
            n_images = len(data["matrices"])
        if n_images < 4:
            parser.error(
                f"At least four images required for half/full budgets: {dataset}"
            )
        symmetry = (
            "D7"
            if dataset.stem.startswith("proteasome")
            else "C4"
            if dataset.stem.startswith("trpv1")
            else "C1"
        )
        for budget, n_third in (("half", n_images // 2), ("full", n_images - 2)):
            for seed in args.seeds:
                for method in ("averaging", "voting"):
                    for handedness in ("star", "synchronization"):
                        prefix = (
                            output
                            / "pipeline"
                            / f"{dataset.stem}_{budget}_seed{seed}_{method}_{handedness}"
                        )
                        prefix.parent.mkdir(parents=True, exist_ok=True)
                        command = [
                            sys.executable,
                            "-m",
                            "scripts.pipeline_errors",
                            str(dataset),
                            "--n-images",
                            str(n_images),
                            "--n-third",
                            str(n_third),
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
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files + sources
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
                    r["budget"],
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
