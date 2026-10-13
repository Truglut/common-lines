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

    monkeypatch.setattr(
        main, "get_transformation_for_second_image", fail_triplet_estimation
    )
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
                result.graph[i, j] @ common_line_i, common_line_j, atol=1.0e-12
            )


def test_simple_api_returns_only_graph(monkeypatch):
    expected_graph = np.broadcast_to(np.eye(3), (3, 3, 3, 3)).copy()
    detailed = main.OrientationEstimationResult(
        graph=expected_graph,
        used_fallback=np.zeros((3, 3), dtype=bool),
        handedness_reference=(0, 1),
    )

    monkeypatch.setattr(
        main,
        "build_relative_orientations_graph_with_diagnostics",
        lambda *args, **kwargs: detailed,
    )

    graph = main.build_relative_orientations_graph(np.zeros((3, 5, 5)))

    assert isinstance(graph, np.ndarray)
    assert graph is expected_graph


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


def test_orientation_graph_uses_i_to_j_coordinate_convention():
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

    candidates, used_fallback = main._estimate_relative_orientation_candidates(
        angle_1,
        angle_2,
        n_third=1,
        random_seed=0,
    )

    assert not np.any(used_fallback)

    graph = main._select_consistent_handedness(
        candidates,
        reference_pair=(0, 1),
    )

    for i in range(3):
        for j in range(3):
            expected = frames[j] @ frames[i].T
            assert np.allclose(graph[i, j], expected, atol=1.0e-12)


def test_fallback_and_reconstitution_edges_share_i_to_j_convention(monkeypatch):
    """Fallback and reconstructed edges must use the same graph convention."""

    # Non-degenerate common-line geometry already used by the alignment tests.
    angle_1 = np.array(
        [
            [0.0, 20.0, 110.0],
            [70.0, 0.0, 145.0],
            [35.0, 125.0, 0.0],
        ]
    )
    angle_2 = np.array(
        [
            [0.0, 70.0, 35.0],
            [20.0, 0.0, 125.0],
            [110.0, 145.0, 0.0],
        ]
    )

    real_solver = main.get_transformation_for_second_image

    def solver_with_one_forced_failure(**kwargs):
        # Force only the directed edge 0 -> 1 to use the pairwise fallback.
        if (
            kwargs["angle_12_in_1_deg"] == 20.0
            and kwargs["angle_12_in_2_deg"] == 70.0
        ):
            raise ValueError("Force fallback for edge 0 -> 1")

        return real_solver(**kwargs)

    monkeypatch.setattr(
        main,
        "get_transformation_for_second_image",
        solver_with_one_forced_failure,
    )

    candidates, used_fallback = main._estimate_relative_orientation_candidates(
        angle_1,
        angle_2,
        n_third=1,
        random_seed=0,
    )

    expected_fallback = np.zeros((3, 3), dtype=bool)
    expected_fallback[0, 1] = True
    assert np.array_equal(used_fallback, expected_fallback)

    reference_pair = main._resolve_handedness_reference(
        used_fallback,
        requested=None,
    )
    assert not used_fallback[reference_pair]

    graph = main._select_consistent_handedness(
        candidates,
        reference_pair=reference_pair,
    )

    for i in range(3):
        for j in range(3):
            if i == j:
                continue

            common_line_i = _image_plane_direction(angle_1[i, j])
            common_line_j = _image_plane_direction(angle_2[i, j])

            assert np.allclose(
                graph[i, j] @ common_line_i,
                common_line_j,
                atol=1.0e-12,
            )


def test_pipeline_resolves_default_preprocessing_once_and_reuses_it(monkeypatch):
    """The same resolved defaults are used for centering and detection."""

    images = np.zeros((3, 5, 5), dtype=np.float64)
    centered_images = np.full_like(images, 2.0)
    detection_images = np.full_like(images, 3.0)
    options = main.PreprocessingOptions(
        mask=np.ones((5, 5), dtype=np.float64),
        subtract_background=False,
        clip_negative=True,
    )

    calls = []

    def fake_precenter(input_images, *, method, preprocessing_options):
        calls.append(("precenter", input_images, preprocessing_options))
        assert input_images is images
        assert method == "center_of_mass"
        return centered_images

    def fake_preprocess(input_images, *, options):
        calls.append(("preprocess", input_images, options))
        assert input_images is centered_images
        return detection_images

    def fake_sinograms(input_images, *, theta):
        calls.append(("sinograms", input_images, None))
        assert input_images is detection_images
        return np.zeros((3, len(theta), 5), dtype=np.float64)

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

    monkeypatch.setattr(main, "precenter_images", fake_precenter)
    monkeypatch.setattr(main, "preprocess_images", fake_preprocess)
    monkeypatch.setattr(main, "batch_common_line_sinograms", fake_sinograms)
    monkeypatch.setattr(
        main,
        "detect_common_lines_all_pairs",
        lambda sinograms, allow_reversed, block_size: FakeCommonLines(),
    )
    monkeypatch.setattr(
        main,
        "get_transformation_for_second_image",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("force fallback")),
    )

    main.build_relative_orientations_graph(
        images,
        theta=np.array([0.0, 90.0]),
        preprocessing_options=options,
    )

    assert [name for name, _, _ in calls] == ["precenter", "preprocess", "sinograms"]
    assert calls[0][2] is options
    assert calls[1][2] is options
