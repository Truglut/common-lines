import numpy as np

import common_lines.main as main


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

    direct, mirrored = main._estimate_pairwise_fallback(theta_1_deg, theta_2_deg)

    _assert_rotation_matrix(direct)
    assert np.allclose(mirrored, direct)

    common_line_1 = _image_plane_direction(theta_1_deg)
    common_line_2 = _image_plane_direction(theta_2_deg)
    assert np.allclose(direct @ common_line_2, common_line_1, atol=1.0e-12)

    delta_rad = np.deg2rad(theta_1_deg - theta_2_deg)
    expected = np.array(
        [
            [np.cos(delta_rad), -np.sin(delta_rad), 0.0],
            [np.sin(delta_rad), np.cos(delta_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(direct, expected, atol=1.0e-12)


def _patch_pipeline_to_force_fallback(monkeypatch):
    angle_1 = np.array(
        [
            [0.0, 10.0, 25.0],
            [40.0, 0.0, 60.0],
            [80.0, 110.0, 0.0],
        ]
    )
    angle_2 = np.array(
        [
            [0.0, 40.0, 80.0],
            [10.0, 0.0, 110.0],
            [25.0, 60.0, 0.0],
        ]
    )

    class FakeCommonLines:
        def oriented_angles(self, theta):
            return angle_1, angle_2

    monkeypatch.setattr(
        main,
        "batch_common_line_sinograms",
        lambda images, theta: np.empty((images.shape[0], len(theta), images.shape[-1])),
    )
    monkeypatch.setattr(
        main,
        "detect_common_lines_all_pairs",
        lambda sinograms, allow_reversed, block_size: FakeCommonLines(),
    )

    def fail_triplet_estimation(**kwargs):
        raise ValueError("No valid triplet geometry")

    monkeypatch.setattr(main, "get_alignment_for_second_image", fail_triplet_estimation)
    return angle_1, angle_2


def test_pipeline_diagnostics_report_fallback_edges(monkeypatch):
    angle_1, angle_2 = _patch_pipeline_to_force_fallback(monkeypatch)

    images = np.zeros((3, 5, 5), dtype=np.float64)
    result = main.build_relative_orientations_graph_with_diagnostics(
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
                result.graph[i, j] @ common_line_j,
                common_line_i,
                atol=1.0e-12,
            )


def test_automatic_handedness_reference_avoids_fallback_edge():
    used_fallback = np.array(
        [
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ]
    )

    reference = main._resolve_handedness_reference(used_fallback, requested=None)

    assert reference == (0, 2)
    assert not used_fallback[reference]
