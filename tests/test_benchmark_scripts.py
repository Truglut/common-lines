"""Tests for image-preparation variants and whole-graph validation rules."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._benchmark_preprocessing import (
    add_preprocessing_arguments,
    prepare_images,
    resolve_preprocessing_options,
)
from scripts.pipeline_errors import (
    evaluate_graph,
    largest_component_size,
    rotation_error_deg,
    valid_rotation_mask,
)
from common_lines.centering import precenter_images
from common_lines.centering.preprocessing import (
    default_preprocessing_options,
    preprocess_images,
)


def rotations():
    a = np.deg2rad(32)
    b = np.deg2rad(51)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return np.stack([np.eye(3), rz, rx, rz @ rx])


def true_graph(matrices):
    return matrices[None, :, :, :] @ matrices[:, None, :, :].swapaxes(-1, -2)


def test_graph_scoring_exact_and_globally_mirrored():
    rs = rotations()
    graph = true_graph(rs)
    fallback = np.zeros(graph.shape[:2], dtype=bool)
    result = evaluate_graph(graph, fallback, rs)
    assert not result["used_mirrored_branch"]
    assert np.nanmax(result["errors"]) < 1e-4
    assert np.nanmax(result["inverse_disagreement"]) < 1e-4

    j = np.diag([1.0, 1.0, -1.0])
    mirrored = j @ graph @ j
    opposite = evaluate_graph(mirrored, fallback, rs)
    assert opposite["used_mirrored_branch"]
    assert np.nanmax(opposite["errors"]) < 1e-4


def test_handedness_is_not_chosen_separately_for_each_edge():
    rs = rotations()
    graph = true_graph(rs)
    j = np.diag([1.0, 1.0, -1.0])
    graph[0, 2] = j @ graph[0, 2] @ j
    result = evaluate_graph(graph, np.zeros(graph.shape[:2], bool), rs)
    assert result["errors"][0, 2] > 1.0
    assert result["errors"][2, 3] < 1e-4


def test_invalid_rotation_is_reported_without_projection():
    rs = rotations()
    graph = true_graph(rs)
    graph[0, 1] *= 1.1
    result = evaluate_graph(graph, np.zeros(graph.shape[:2], bool), rs)
    assert not result["valid"][0, 1]
    assert np.isnan(result["errors"][0, 1])
    assert result["valid"][1, 0]
    assert result["residual"][0, 1] > 0.1


def test_relative_rotations_use_direction_i_to_j():
    rs = rotations()
    graph = true_graph(rs)
    assert rotation_error_deg(graph[0, 2], rs[2] @ rs[0].T) < 1e-5
    assert rotation_error_deg(graph[2, 0], rs[0] @ rs[2].T) < 1e-5
    assert np.all(valid_rotation_mask(graph)[0])


def test_nonfallback_coverage_component():
    n = 5
    fallback = np.ones((n, n), bool)
    np.fill_diagonal(fallback, False)
    fallback[0, 1] = False
    fallback[1, 2] = False
    fallback[3, 4] = False
    assert largest_component_size(fallback, np.ones((n, n), bool)) == 3
    assert largest_component_size(np.ones((n, n), bool), np.ones((n, n), bool)) == 1


def options_for(args, *, pipeline=False):
    parser = argparse.ArgumentParser()
    add_preprocessing_arguments(parser, pipeline_defaults=pipeline)
    return parser.parse_args(args)


def test_default_preprocessing_matches_pipeline_and_raw_mode():
    images = np.random.default_rng(2).normal(size=(4, 33, 33))
    raw = resolve_preprocessing_options(images, options_for([]))
    np.testing.assert_allclose(
        prepare_images(images, precenter="none", preprocessing_options=raw), images
    )
    default = resolve_preprocessing_options(images, options_for([], pipeline=True))
    expected = default_preprocessing_options(images)
    np.testing.assert_array_equal(default.mask, expected.mask)
    assert default.subtract_background == expected.subtract_background
    np.testing.assert_allclose(
        prepare_images(
            images, precenter="center_of_mass", preprocessing_options=default
        ),
        preprocess_images(
            precenter_images(
                images, method="center_of_mass", preprocessing_options=default
            ),
            options=default,
        ),
    )


def test_mask_background_and_clip_can_be_switched_independently():
    images = np.ones((3, 17, 17))
    custom = options_for(["--preprocessing", "default", "--no-mask", "--clip-negative"])
    options = resolve_preprocessing_options(images, custom)
    assert options.mask is None
    assert options.background_mask is not None
    assert options.subtract_background
    assert options.clip_negative

    no_bg = options_for(["--preprocessing", "default", "--no-subtract-background"])
    options = resolve_preprocessing_options(images, no_bg)
    assert options.mask is not None
    assert not options.subtract_background
    assert options.background_mask is None

    radius = options_for(["--mask", "--mask-radius-fraction", "0.7"])
    options = resolve_preprocessing_options(images, radius)
    assert options.mask.sum() < default_preprocessing_options(images).mask.sum()


def test_public_pipeline_with_exact_detected_common_lines(monkeypatch):
    """Validate the full public graph path without relying on image interpolation."""
    import common_lines.main as main
    from scripts.orientation_errors import ground_truth_angles

    rng = np.random.default_rng(7)
    rs = []
    for _ in range(6):
        u, _, vt = np.linalg.svd(rng.normal(size=(3, 3)))
        rs.append(u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt)
    matrices = np.stack(rs)
    first, second, _ = ground_truth_angles(matrices)
    i, j = np.triu_indices(len(matrices), k=1)
    first[j, i] = second[i, j]
    second[j, i] = first[i, j]

    class FakeCommonLines:
        def oriented_angles(self, theta):
            return first, second

    monkeypatch.setattr(
        main,
        "batch_common_line_sinograms",
        lambda images, theta: np.zeros((len(images), 9, len(theta))),
    )
    monkeypatch.setattr(
        main,
        "detect_common_lines_all_pairs",
        lambda sinograms, allow_reversed, block_size: FakeCommonLines(),
    )
    estimate = main.build_relative_orientations_graph_with_diagnostics(
        np.ones((6, 9, 9)),
        theta=np.arange(0.0, 180.0, 15.0),
        n_third=2,
        precenter_method=None,
        preprocessing_options=main.PreprocessingOptions(),
    )
    assert not estimate.used_fallback.any()
    scored = evaluate_graph(estimate.graph, estimate.used_fallback, matrices)
    assert np.nanmax(scored["errors"]) < 1e-4
