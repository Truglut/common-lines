"""Symmetry-aware local and graph validation: exact SO(3) gauge relationships."""
import sys
from pathlib import Path

import numpy as np

from common_lines import mirror_relative_orientations_graph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.detection_errors import symmetry_operators
from scripts.orientation_errors import (
    evaluate_triplet,
    ground_truth_angles,
    symmetry_triplet_errors,
)
from scripts.pipeline_errors import evaluate_graph


def rotations(n=6, seed=303):
    rng = np.random.default_rng(seed)
    u, _, vt = np.linalg.svd(rng.normal(size=(n, 3, 3)))
    det = np.linalg.det(u @ vt)
    diag = np.zeros((n, 3, 3))
    diag[:, range(3), range(3)] = 1
    diag[:, 2, 2] = det
    return u @ diag @ vt


def graph_from(matrices):
    return matrices[None, :, :, :] @ matrices[:, None, :, :].swapaxes(-1, -2)


def test_c1_matches_no_symmetry_default_for_triplet_and_graph():
    rs = rotations()
    a, b, _ = ground_truth_angles(rs)
    corr = np.full_like(a, np.nan)
    base = evaluate_triplet(0, 1, 2, a, b, corr, rs)
    explicit = evaluate_triplet(0, 1, 2, a, b, corr, rs, symmetry_operators("C1"))
    assert base["status"] == explicit["status"] == "ok"
    assert base["mean_rotation_error_deg"] == explicit["mean_rotation_error_deg"]
    graph = graph_from(rs)
    mask = np.zeros((len(rs), len(rs)), dtype=bool)
    baseline = evaluate_graph(graph, mask, rs)
    c1 = evaluate_graph(graph, mask, rs, symmetry_operators("C1"))
    np.testing.assert_array_equal(baseline["errors"], c1["errors"])


def test_graph_symmetry_consistent_independent_orientation_gauges():
    rs = rotations()
    for symmetry in ("C5", "D7"):
        ops = symmetry_operators(symmetry)
        gauge = ops[np.arange(len(rs)) % len(ops)]
        graph = graph_from(rs @ gauge)
        fallback = np.zeros((len(rs), len(rs)), bool)
        result = evaluate_graph(graph, fallback, rs, ops)
        assert np.nanmax(result["errors"]) < 1e-4
        assert np.nanmax(result["inverse_disagreement"]) < 1e-4
        assert not result["used_mirrored_branch"]
        c1_result = evaluate_graph(graph, fallback, rs)
        assert np.nanmean(c1_result["errors"]) > 3.0
        mirrored = evaluate_graph(
            mirror_relative_orientations_graph(graph), fallback, rs, ops
        )
        assert mirrored["used_mirrored_branch"]
        assert np.nanmax(mirrored["errors"]) < 1e-4


def test_triplet_symmetry_gauge_consistency_and_independent_edge_trap():
    rs = rotations(3)
    ops = symmetry_operators("D7")
    sj, sk = ops[2], ops[8]
    ri, rj, rk = rs
    ij = rj @ sj @ ri.T
    ik = rk @ sk @ ri.T
    jk = rk @ sk @ sj.T @ rj.T
    errors, j_idx, k_idx = symmetry_triplet_errors((ij, ik, jk), rs, ops)
    assert np.max(errors) < 1e-4
    assert j_idx == 2 and k_idx == 8

    # Each edge separately has a perfect symmetry match, but the three
    # relative rotations cannot all arise from the same image gauges.
    inconsistent_jk = rk @ ops[0] @ rj.T
    errors, _, _ = symmetry_triplet_errors((ij, ik, inconsistent_jk), rs, ops)
    assert np.mean(errors) > 1.0


def test_exact_triplet_scoring_for_cyclic_dihedral_and_reference_gauge():
    rs = rotations()
    a, b, _ = ground_truth_angles(rs)
    corrs = np.full_like(a, np.nan)
    for symmetry in ("C1", "C7", "D7"):
        ops = symmetry_operators(symmetry)
        score = evaluate_triplet(1, 2, 4, a, b, corrs, rs, ops)
        assert score["status"] == "ok"
        assert score["max_rotation_error_deg"] < 1e-4
        # The same representative common-line measurements remain compatible
        # with ground-truth matrices re-gauged independently by symmetry.
        rs_gauged = rs @ ops[np.arange(len(rs)) % len(ops)]
        score_gauged = evaluate_triplet(1, 2, 4, a, b, corrs, rs_gauged, ops)
        assert score_gauged["status"] == "ok"
        assert score_gauged["max_rotation_error_deg"] < 1e-4


def test_symmetry_validation_does_not_hide_invalid_rotations():
    rs = rotations(4)
    graph = graph_from(rs)
    graph[0, 1] *= 1.05
    result = evaluate_graph(graph, np.zeros((4, 4), bool), rs, symmetry_operators("D7"))
    assert not result["valid"][0, 1]
    assert np.isnan(result["errors"][0, 1])
    assert result["symmetry_operator_index"][0, 1] == -1
    assert result["valid"][1, 0]
