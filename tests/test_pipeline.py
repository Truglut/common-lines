from typing import Literal, get_args
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common_lines import pipeline
from common_lines.geometry.forward import compute_intrinsic_common_line_angles
from common_lines.graph_estimation import HandednessMethod
from common_lines.graph_estimation import handedness as graph_handedness


def _image_plane_direction(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    return np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])


def _assert_rotation_matrix(rotation: np.ndarray, *, atol: float = 1.0e-12) -> None:
    assert rotation.shape == (3, 3)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=atol)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=atol)


def _patch_detected_common_lines(
    monkeypatch: pytest.MonkeyPatch, angle_1: np.ndarray, angle_2: np.ndarray
) -> None:
    """Supply detected angles in degrees while running real graph estimation."""

    class FakeCommonLines:
        def oriented_angles(
            self, theta: np.ndarray, radians: bool = False
        ) -> tuple[np.ndarray, np.ndarray]:
            assert radians is True
            return np.deg2rad(angle_1), np.deg2rad(angle_2)

    monkeypatch.setattr(
        pipeline,
        "batch_common_line_sinograms",
        lambda images, theta: np.empty((images.shape[0], len(theta), images.shape[-1])),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_common_lines_all_pairs",
        lambda sinograms, allow_reversed, block_size: FakeCommonLines(),
    )


def _patch_pipeline_to_force_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[np.ndarray, np.ndarray]:
    # Every common line in an image has the same direction, so all triplets
    # are degenerate. Distinct image angles exercise nonidentity fallbacks.
    angle_1 = np.array([[0.0, 10.0, 10.0], [40.0, 0.0, 40.0], [80.0, 80.0, 0.0]])
    angle_2 = angle_1.T.copy()
    _patch_detected_common_lines(monkeypatch, angle_1, angle_2)
    return angle_1, angle_2


def test_pipeline_diagnostics_report_fallback_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    angle_1, angle_2 = _patch_pipeline_to_force_fallback(monkeypatch)

    images = np.zeros((3, 5, 5), dtype=np.float64)
    result = pipeline.build_relative_orientations_graph_with_diagnostics(
        images,
        theta=np.array([0.0, 90.0]),
        n_third=1,
        precenter_method=None,
    )

    expected_fallback_mask = ~np.eye(3, dtype=bool)
    assert np.array_equal(result.used_fallback, expected_fallback_mask)
    assert result.handedness_reference == (0, 1)

    for i in range(3):
        assert np.allclose(result.graph[i, i], np.eye(3))

        for j in range(3):
            if i == j:
                continue

            _assert_rotation_matrix(result.graph[i, j])

            common_line_i = _image_plane_direction(angle_1[i, j])
            common_line_j = _image_plane_direction(angle_2[i, j])

            assert np.allclose(
                result.graph[i, j] @ common_line_i, common_line_j, atol=1.0e-12
            )


def test_simple_api_returns_only_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline_to_force_fallback(monkeypatch)
    images = np.zeros((3, 5, 5))
    theta = np.array([0.0, 90.0])
    detailed = pipeline.build_relative_orientations_graph_with_diagnostics(
        images, theta=theta
    )
    graph = pipeline.build_relative_orientations_graph(images, theta=theta)

    assert isinstance(graph, np.ndarray)
    np.testing.assert_array_equal(graph, detailed.graph)


@pytest.mark.parametrize("handedness_method", get_args(HandednessMethod))
def test_fallback_and_reconstitution_edges_share_i_to_j_convention(
    monkeypatch: pytest.MonkeyPatch,
    handedness_method: HandednessMethod,
) -> None:
    """Fallback and reconstructed edges must use the same graph convention."""
    frames = Rotation.from_euler(
        "ZYZ", [[0, 0, 0], [0.3, 0.7, -0.2], [0.8, 1.2, 0.4], [-0.8, 0.5, 0.3]]
    ).as_matrix()
    first, second = compute_intrinsic_common_line_angles(
        frames[:, None], frames[None, :]
    )
    # Make all lines in image 0 parallel. Edges involving image 0 must fall
    # back, while images 1, 2, 3 still have a valid reconstruction triplet.
    first[0, 1:] = np.deg2rad(20.0)
    second[1:, 0] = np.deg2rad(20.0)
    angle_1, angle_2 = np.rad2deg(first), np.rad2deg(second)
    _patch_detected_common_lines(monkeypatch, angle_1, angle_2)

    result = pipeline.build_relative_orientations_graph_with_diagnostics(
        np.zeros((4, 5, 5)),
        theta=np.array([0.0, 90.0]),
        n_third=2,
        random_seed=4,
        handedness_method=handedness_method,
    )

    expected_fallback = np.zeros((4, 4), dtype=bool)
    expected_fallback[0, 1:] = True
    expected_fallback[1:, 0] = True
    np.testing.assert_array_equal(result.used_fallback, expected_fallback)
    assert result.handedness_reference == (1, 2)
    assert not result.used_fallback[result.handedness_reference]

    for i in range(4):
        for j in range(4):
            _assert_rotation_matrix(result.graph[i, j])
            if i == j:
                np.testing.assert_allclose(result.graph[i, j], np.eye(3))
                continue

            common_line_i = _image_plane_direction(angle_1[i, j])
            common_line_j = _image_plane_direction(angle_2[i, j])
            np.testing.assert_allclose(
                result.graph[i, j] @ common_line_i,
                common_line_j,
                rtol=0,
                atol=1.0e-12,
            )


@pytest.mark.parametrize("precenter_method", [None, "center_of_mass", "reflection"])
@pytest.mark.parametrize("enabled", [False, True])
def test_pipeline_preprocessing_order(monkeypatch, precenter_method, enabled):
    _patch_pipeline_to_force_fallback(monkeypatch)

    images = np.arange(75, dtype=np.float64).reshape(3, 5, 5) - 30.0
    original = images.copy()
    centered = images + 10.0
    shifts = np.zeros((3, 2))

    options = pipeline.PreprocessingOptions(clip_negative=True) if enabled else None
    real_preprocess = pipeline.preprocess_images
    calls = []
    processed_batches = []

    expected_options = options
    expected_method = precenter_method

    def preprocess(batch, *, options):
        assert options is expected_options
        expected = images if not processed_batches else centered
        assert batch is expected

        result = real_preprocess(batch, options=options)
        calls.append("preprocess")
        processed_batches.append(result)
        return result

    def estimate(batch, *, method):
        assert method == expected_method
        assert batch is processed_batches[0]
        calls.append("estimate")
        return shifts

    def shift(batch, corrections):
        assert batch is images
        assert corrections is shifts
        calls.append("shift")
        return centered

    def sinograms(batch, *, theta):
        assert batch is processed_batches[-1]

        expected = centered if precenter_method is not None else images
        np.testing.assert_array_equal(
            batch,
            real_preprocess(expected, options=options),
        )

        calls.append("sinograms")
        return np.zeros((3, len(theta), 5))

    monkeypatch.setattr(pipeline, "preprocess_images", preprocess)
    monkeypatch.setattr(pipeline, "estimate_shifts", estimate)
    monkeypatch.setattr(pipeline, "shift_images", shift)
    monkeypatch.setattr(
        pipeline,
        "batch_common_line_sinograms",
        sinograms,
    )

    # Omit disabled options so these cases also exercise the defaults.
    kwargs = {}
    if precenter_method is not None:
        kwargs["precenter_method"] = precenter_method
    if enabled:
        kwargs["preprocessing_options"] = options

    pipeline.build_relative_orientations_graph(
        images,
        theta=np.array([0.0, 90.0]),
        **kwargs,
    )

    expected_calls = (
        ["preprocess", "estimate", "shift"] if precenter_method is not None else []
    )
    assert calls == expected_calls + ["preprocess", "sinograms"]
    np.testing.assert_array_equal(images, original)


@pytest.mark.parametrize("api", ["simple", "diagnostics"])
@pytest.mark.parametrize("handedness_method", [None, *get_args(HandednessMethod)])
def test_pipeline_dispatches_requested_handedness_method(
    monkeypatch: pytest.MonkeyPatch,
    api: Literal["simple", "diagnostics"],
    handedness_method: HandednessMethod | None,
) -> None:
    _patch_pipeline_to_force_fallback(monkeypatch)
    builder = (
        pipeline.build_relative_orientations_graph
        if api == "simple"
        else pipeline.build_relative_orientations_graph_with_diagnostics
    )
    kwargs: dict[str, HandednessMethod] = {}
    if handedness_method is not None:
        kwargs["handedness_method"] = handedness_method

    # Spy on the real algorithms: equal outputs on simple graphs would hide a
    # dropped method argument or an incorrect selector branch.
    with (
        patch.object(
            graph_handedness,
            "_synchronize_with_star",
            wraps=graph_handedness._synchronize_with_star,
        ) as star,
        patch.object(
            graph_handedness,
            "_synchronize_with_j_sync",
            wraps=graph_handedness._synchronize_with_j_sync,
        ) as synchronization,
    ):
        result = builder(
            np.zeros((3, 5, 5)),
            theta=np.array([0.0, 90.0]),
            handedness_reference=(2, 0),
            **kwargs,
        )
        selected, other = (
            (synchronization, star)
            if handedness_method == "synchronization"
            else (star, synchronization)
        )
        selected.assert_called_once()
        other.assert_not_called()
        assert selected.call_args.kwargs["reference_pair"] == (2, 0)

    if api == "diagnostics":
        assert isinstance(result, pipeline.OrientationEstimationResult)
        assert result.handedness_reference == (2, 0)
    else:
        assert isinstance(result, np.ndarray)
