"""Regression tests for scripts/orientation_errors.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common_lines.reconstitution.triplets import HANDEDNESS_REFLECTION
from scripts.orientation_errors import (
    evaluate_triplet,
    ground_truth_angles,
    perturb_angles,
    rotation_error_deg,
    sample_triplets,
)


def random_rotations(n=10, seed=12):
    rng = np.random.default_rng(seed)
    rotations = []
    for _ in range(n):
        m = rng.normal(size=(3, 3))
        u, _, vt = np.linalg.svd(m)
        d = np.diag([1, 1, np.linalg.det(u @ vt)])
        rotations.append(u @ d @ vt)
    return np.stack(rotations)


def test_exact_common_lines_reconstruct_relative_rotations():
    matrices = random_rotations()
    first, second, _ = ground_truth_angles(matrices)
    for i, j, k in sample_triplets(matrices, 80, np.random.default_rng(0)):
        result = evaluate_triplet(
            i, j, k, first, second, np.full_like(first, np.nan), matrices
        )
        assert result["status"] == "ok"
        assert result["max_rotation_error_deg"] < 1e-4


def test_mirrored_ground_truth_has_same_best_triplet_error():
    matrices = random_rotations()
    mirror = HANDEDNESS_REFLECTION
    mirrored = mirror @ matrices @ mirror
    first, second, _ = ground_truth_angles(matrices)
    r = evaluate_triplet(0, 1, 2, first, second, np.full_like(first, np.nan), matrices)
    mirrored_result = evaluate_triplet(
        0, 1, 2, first, second, np.full_like(first, np.nan), mirrored
    )
    assert r["status"] == mirrored_result["status"] == "ok"
    assert r["max_rotation_error_deg"] < 1e-4
    assert mirrored_result["max_rotation_error_deg"] < 1e-4


def test_zero_perturbation_matches_exact_and_sampling_is_reproducible():
    matrices = random_rotations()
    first, second, _ = ground_truth_angles(matrices)
    a, b = perturb_angles(first, second, 0.0, np.random.default_rng(4))
    np.testing.assert_allclose(a, first, equal_nan=True, atol=1e-12)
    np.testing.assert_allclose(b, second, equal_nan=True, atol=1e-12)
    np.testing.assert_array_equal(
        sample_triplets(matrices, 12, np.random.default_rng(1)),
        sample_triplets(matrices, 12, np.random.default_rng(1)),
    )


def test_noisy_angles_keep_finite_rotation_errors_when_solvable():
    matrices = random_rotations(12)
    first, second, _ = ground_truth_angles(matrices)
    a, b = perturb_angles(first, second, 2.0, np.random.default_rng(44))
    rows = [
        evaluate_triplet(i, j, k, a, b, np.full_like(first, np.nan), matrices)
        for i, j, k in sample_triplets(matrices, 40, np.random.default_rng(8))
    ]
    successful = [row for row in rows if row["status"] == "ok"]
    assert successful
    assert all(np.isfinite(r["max_rotation_error_deg"]) for r in successful)
    assert any(r["mean_rotation_error_deg"] > 0.1 for r in successful)


def test_geodesic_rotation_error():
    theta = np.deg2rad(36)
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    assert abs(rotation_error_deg(np.eye(3), rot) - 36.0) < 1e-9
