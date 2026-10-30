"""Tests for benchmark image preparation and whole-graph validation rules."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common_lines.centering import estimate_shifts, shift_images
from common_lines.preprocessing import (
    PreprocessingOptions,
    preprocess_images,
)
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


def options_for(args):
    parser = argparse.ArgumentParser()
    add_preprocessing_arguments(parser)
    return parser.parse_args(args)


def test_default_preprocessing_disabled_and_opt_in_matches_pipeline():
    images = np.random.default_rng(2).normal(size=(4, 33, 33))
    args = options_for([])
    assert args.precenter == "none" and args.preprocessing == "none"
    assert resolve_preprocessing_options(images, args) is None
    np.testing.assert_allclose(
        prepare_images(images, precenter=args.precenter, preprocessing_options=None),
        images,
    )

    default = resolve_preprocessing_options(
        images, options_for(["--preprocessing", "default"])
    )
    expected = PreprocessingOptions.default_from_images(images)
    np.testing.assert_array_equal(default.mask, expected.mask)
    assert default.subtract_background == expected.subtract_background
    assert default.clip_negative == expected.clip_negative
    assert default.gaussian_lowpass_sigma is None
    np.testing.assert_allclose(
        prepare_images(
            images, precenter="center_of_mass", preprocessing_options=default
        ),
        preprocess_images(
            shift_images(
                images,
                estimate_shifts(
                    preprocess_images(images, options=default), method="center_of_mass"
                ),
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
    assert (
        options.mask.sum() < PreprocessingOptions.default_from_images(images).mask.sum()
    )


def test_gaussian_filter_without_other_preprocessing():
    images = np.zeros((2, 25, 25))
    images[:, 12, 12] = 1.0
    args = options_for(["--gaussian-lowpass-sigma", "1.5"])
    options = resolve_preprocessing_options(images, args)
    assert options is not None
    assert options.mask is None
    assert not options.subtract_background
    assert options.gaussian_lowpass_sigma == 1.5
    filtered = prepare_images(images, precenter="none", preprocessing_options=options)
    np.testing.assert_allclose(filtered, preprocess_images(images, options=options))
    assert filtered[0, 12, 12] < 1.0
    assert np.isclose(filtered.sum(), images.sum())
    for value in ("0", "-1", "nan", "inf"):
        with pytest.raises(ValueError, match="gaussian-lowpass-sigma"):
            resolve_preprocessing_options(
                images, options_for(["--gaussian-lowpass-sigma", value])
            )


def test_gaussian_preprocessing_used_for_shift_estimation_but_shifts_apply_to_original(
    monkeypatch,
):
    import scripts._benchmark_preprocessing as prep

    images = np.random.default_rng(5).normal(size=(2, 17, 17))
    opts = resolve_preprocessing_options(
        images, options_for(["--gaussian-lowpass-sigma", "2", "--clip-negative"])
    )
    supplied = []

    def fake_estimate(images_for_centers, *, method):
        supplied.append((images_for_centers.copy(), method))
        return np.array([[0.0, 1.0], [1.0, 0.0]])

    monkeypatch.setattr(prep, "estimate_shifts", fake_estimate)
    output = prepare_images(images, precenter="reflection", preprocessing_options=opts)
    np.testing.assert_allclose(supplied[0][0], preprocess_images(images, options=opts))
    assert supplied[0][1] == "reflection"
    np.testing.assert_allclose(
        output,
        preprocess_images(
            shift_images(images, np.array([[0, 1], [1, 0]])), options=opts
        ),
    )


def test_shared_preparation_matches_public_pipeline(monkeypatch):
    """Capture images entering the Radon transform in the production pipeline."""
    import common_lines.pipeline as main

    rng = np.random.default_rng(17)
    images = rng.random((3, 21, 21))
    args = options_for(
        [
            "--precenter",
            "center_of_mass",
            "--preprocessing",
            "default",
            "--gaussian-lowpass-sigma",
            "1.25",
            "--no-mask",
        ]
    )
    options = resolve_preprocessing_options(images, args)
    expected = prepare_images(
        images, precenter=args.precenter, preprocessing_options=options
    )
    captured = []

    def capture_sinograms(prepared, theta):
        captured.append(prepared.copy())
        return np.zeros((len(prepared), 9, len(theta)))

    class FakeCommonLines:
        def oriented_angles(
            self, theta: np.ndarray, radians: bool = False
        ) -> tuple[np.ndarray, np.ndarray]:
            assert radians is True
            n = len(images)
            return np.zeros((n, n)), np.zeros((n, n))

    monkeypatch.setattr(main, "batch_common_line_sinograms", capture_sinograms)
    monkeypatch.setattr(
        main,
        "detect_common_lines_all_pairs",
        lambda sinograms, allow_reversed, block_size: FakeCommonLines(),
    )
    main.build_relative_orientations_graph_with_diagnostics(
        images,
        n_third=1,
        theta=np.arange(0, 180, 30),
        precenter_method=args.precenter,
        preprocessing_options=options,
    )
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0], expected)


def test_public_pipeline_with_exact_detected_common_lines(monkeypatch):
    """Validate the full public graph path without relying on image interpolation."""
    import common_lines.pipeline as main
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
        def oriented_angles(
            self, theta: np.ndarray, radians: bool = False
        ) -> tuple[np.ndarray, np.ndarray]:
            assert radians is True
            return np.deg2rad(first), np.deg2rad(second)

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
    )
    assert not estimate.used_fallback.any()
    scored = evaluate_graph(estimate.graph, estimate.used_fallback, matrices)
    assert np.nanmax(scored["errors"]) < 1e-4


@pytest.mark.parametrize("method", ["averaging", "voting"])
def test_pipeline_cli_passes_and_records_reconstitution_method(
    tmp_path, monkeypatch, method
):
    import json
    from types import SimpleNamespace

    from scripts import pipeline_errors

    matrices = rotations()
    metadata = tmp_path / "particles.npz"
    np.savez(metadata, matrices=matrices, original_images=np.ones((4, 9, 9)))
    prefix = tmp_path / method

    def estimate(images, **kwargs):
        assert kwargs["reconstitution_method"] == method
        return SimpleNamespace(
            graph=true_graph(matrices),
            used_fallback=np.zeros((4, 4), dtype=bool),
            handedness_reference=(0, 1),
        )

    monkeypatch.setattr(
        pipeline_errors, "build_relative_orientations_graph_with_diagnostics", estimate
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline_errors",
            str(metadata),
            "--reconstitution-method",
            method,
            "--output",
            str(prefix),
        ],
    )
    pipeline_errors.main()
    with np.load(prefix.with_suffix(".npz")) as data:
        configuration = json.loads(str(data["run_configuration_json"]))
    assert configuration["reconstitution_method"] == method
