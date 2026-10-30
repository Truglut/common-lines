from typing import get_args

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import common_lines.graph_estimation
from common_lines import pipeline
from common_lines.forward import compute_intrinsic_common_line_angles
from src.common_lines.graph_estimation import ReconstitutionMethod


def _image_plane_direction(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    return np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])


def _assert_rotation_matrix(rotation: np.ndarray, *, atol: float = 1.0e-12) -> None:
    assert rotation.shape == (3, 3)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=atol)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=atol)


def test_pairwise_fallback_is_minimal_in_plane_rotation():
    theta_1_deg = 25.0
    theta_2_deg = 100.0

    # With only two images there is no third view for reconstitution.
    candidates, used_fallback = (
        common_lines.graph_estimation.estimate_relative_orientation_candidates(
            np.deg2rad([[0.0, theta_1_deg], [theta_2_deg, 0.0]]),
            np.deg2rad([[0.0, theta_2_deg], [theta_1_deg, 0.0]]),
            n_third=1,
            random_seed=0,
            method="averaging",  # method does not matter with no third image
        )
    )
    np.testing.assert_array_equal(used_fallback, ~np.eye(2, dtype=bool))
    direct, mirrored = candidates[0, 1]

    _assert_rotation_matrix(direct)
    assert np.allclose(mirrored, direct)

    common_line_1 = _image_plane_direction(theta_1_deg)
    common_line_2 = _image_plane_direction(theta_2_deg)

    assert np.allclose(direct @ common_line_1, common_line_2, atol=1.0e-12)

    delta_rad = np.deg2rad(theta_2_deg - theta_1_deg)
    expected = np.array(
        [
            [np.cos(delta_rad), -np.sin(delta_rad), 0.0],
            [np.sin(delta_rad), np.cos(delta_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert np.allclose(direct, expected, atol=1.0e-12)


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


def test_pipeline_diagnostics_report_fallback_edges(monkeypatch):
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


@pytest.mark.parametrize("method", get_args(ReconstitutionMethod))
def test_orientation_graph_uses_i_to_j_coordinate_convention(
    method: ReconstitutionMethod,
):
    """graph[i, j] transforms coordinates from image i to image j."""

    # Three hand-checkable frames. Each R_i maps coordinates from the
    # reference system into the coordinate system of image i.
    R0 = np.eye(3)
    R1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    R2 = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    frames = (R0, R1, R2)

    # Exact oriented common-line angles for these three frames.
    #
    # angle_1[i, j]: angle in image i
    # angle_2[i, j]: corresponding angle in image j
    angle_1 = np.array(
        [
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
            [0.0, 90.0, 0.0],
        ]
    )
    angle_2 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 90.0],
            [90.0, 90.0, 0.0],
        ]
    )

    candidates, used_fallback = (
        common_lines.graph_estimation.estimate_relative_orientation_candidates(
            np.deg2rad(angle_1),
            np.deg2rad(angle_2),
            n_third=1,
            random_seed=0,
            method=method,
        )
    )

    assert not np.any(used_fallback)

    graph = common_lines.graph_estimation.select_consistent_handedness(
        candidates,
        reference_pair=(0, 1),
    )

    for i in range(3):
        for j in range(3):
            expected = frames[j] @ frames[i].T
            assert np.allclose(graph[i, j], expected, atol=1.0e-12)


@pytest.mark.parametrize("random_seed", [0, 4])
def test_fallback_and_reconstitution_edges_share_i_to_j_convention(
    monkeypatch: pytest.MonkeyPatch, random_seed: int
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
        random_seed=random_seed,
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

    expected_calls = ["preprocess", "estimate", "shift"] if precenter_method is not None else []
    assert calls == expected_calls + ["preprocess", "sinograms"]
    np.testing.assert_array_equal(images, original)
