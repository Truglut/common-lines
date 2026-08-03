import numpy as np
import pytest

from common_lines.detection import (
    CommonLineGraph,
    detect_common_lines,
    detect_common_lines_all_pairs_blocked,
)


def test_pairwise_detection_finds_an_inserted_identical_profile() -> None:
    """An exactly copied profile should be selected with correlation one."""
    rng = np.random.default_rng(1)
    sinogram_1 = rng.normal(size=(13, 5))
    sinogram_2 = rng.normal(size=(13, 7))
    sinogram_2[:, 4] = sinogram_1[:, 2]

    result = detect_common_lines(sinogram_1, sinogram_2)

    assert result.index_1.item() == 2
    assert result.index_2.item() == 4
    np.testing.assert_allclose(result.correlation.item(), 1.0)


def test_pairwise_detection_broadcasts_a_single_reference() -> None:
    """Broadcasted batch results should equal independent pairwise calls."""
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(11, 6))
    batch = rng.normal(size=(3, 11, 5))

    result = detect_common_lines(reference, batch)

    for batch_index in range(batch.shape[0]):
        expected = detect_common_lines(reference, batch[batch_index])
        assert result.index_1[batch_index] == expected.index_1.item()
        assert result.index_2[batch_index] == expected.index_2.item()
        np.testing.assert_allclose(
            result.correlation[batch_index],
            expected.correlation.item(),
        )


def test_pairwise_detection_orients_a_reversed_profile() -> None:
    """A detector-reversed match should add 180 degrees to angle_2."""
    rng = np.random.default_rng(3)
    sinogram_1 = rng.normal(size=(15, 5))
    sinogram_2 = rng.normal(size=(15, 6))
    sinogram_2[:, 4] = sinogram_1[::-1, 1]

    theta_1 = np.array([0.0, 20.0, 55.0, 100.0, 150.0])
    theta_2 = np.array([0.0, 10.0, 40.0, 80.0, 130.0, 170.0])

    result = detect_common_lines(
        sinogram_1,
        sinogram_2,
        theta_1=theta_1,
        theta_2=theta_2,
        allow_reversed=True,
    )

    angle_1, angle_2 = result.oriented_angles(theta_1, theta_2)

    assert result.index_1.item() == 1
    assert result.index_2.item() == 4
    assert result.reversed_match.item()
    assert angle_1.item() == theta_1[1]
    assert angle_2.item() == (theta_2[4] + 180.0) % 360.0
    np.testing.assert_allclose(result.correlation.item(), 1.0)


def test_constant_profiles_produce_zero_correlation() -> None:
    """Zero-variance columns are represented safely as zero profiles."""
    result = detect_common_lines(
        np.ones((8, 3)),
        np.ones((8, 4)),
    )

    assert result.index_1.item() == 0
    assert result.index_2.item() == 0
    assert result.correlation.item() == 0.0


def test_blocked_detection_matches_pairwise_reference() -> None:
    """The blocked algorithm should reproduce explicit pairwise detection.

    Random sinograms are used to make accidental ties very unlikely. Every
    off-diagonal result is compared with the simpler pairwise implementation,
    which acts as the numerical reference for different block boundaries.
    """
    rng = np.random.default_rng(5)
    sinograms = rng.normal(size=(5, 13, 9))

    # Testing two different block boundaries and both reversal modes is enough
    # to exercise diagonal, off-diagonal and final partial blocks.
    for block_size, allow_reversed in ((1, False), (2, True), (4, False)):
        result = detect_common_lines_all_pairs_blocked(
            sinograms,
            allow_reversed=allow_reversed,
            block_size=block_size,
        )

        assert isinstance(result, CommonLineGraph)
        assert result.index_1.shape == (5, 5)
        assert result.index_2.shape == (5, 5)
        assert result.correlation.shape == (5, 5)

        for i in range(sinograms.shape[0]):
            for j in range(sinograms.shape[0]):
                if i == j:
                    continue

                expected = detect_common_lines(
                    sinograms[i],
                    sinograms[j],
                    allow_reversed=allow_reversed,
                )

                assert result.index_1[i, j] == expected.index_1.item()
                assert result.index_2[i, j] == expected.index_2.item()
                np.testing.assert_allclose(
                    result.correlation[i, j],
                    expected.correlation.item(),
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )

                if allow_reversed:
                    assert (
                        result.reversed_match[i, j]
                        == expected.reversed_match.item()
                    )


def test_blocked_detection_reuses_the_opposite_pair_correctly() -> None:
    """Transposed storage must exchange indices but preserve the match score."""
    rng = np.random.default_rng(6)
    sinograms = rng.normal(size=(3, 11, 5))
    sinograms[1, :, 3] = sinograms[0, ::-1, 1]

    result = detect_common_lines_all_pairs_blocked(
        sinograms,
        allow_reversed=True,
        block_size=1,
    )

    assert result.index_1[0, 1] == 1
    assert result.index_2[0, 1] == 3
    assert result.index_1[1, 0] == 3
    assert result.index_2[1, 0] == 1
    assert result.reversed_match[0, 1]
    assert result.reversed_match[1, 0]
    np.testing.assert_allclose(result.correlation[0, 1], 1.0)
    np.testing.assert_allclose(result.correlation[1, 0], 1.0)


def test_common_line_graph_materializes_oriented_angles_on_demand() -> None:
    """The compact graph should reconstruct angles without storing them."""
    graph = CommonLineGraph(
        index_1=np.array([[0, 1], [2, 0]], dtype=np.uint8),
        index_2=np.array([[0, 2], [1, 0]], dtype=np.uint8),
        correlation=np.ones((2, 2)),
        reversed_match=np.array([[False, True], [True, False]]),
        n_angles=3,
    )
    theta = np.array([0.0, 30.0, 75.0])

    angle_1, angle_2 = graph.oriented_angles(theta)

    np.testing.assert_array_equal(angle_1, theta[graph.index_1])
    np.testing.assert_array_equal(
        angle_2,
        np.mod(theta[graph.index_2] + 180.0 * graph.reversed_match, 360.0),
    )
    assert not hasattr(graph, "angle_1")
    assert not hasattr(graph, "angle_2")


def test_blocked_indices_use_a_compact_grid_dependent_dtype() -> None:
    """Angular indices should not be stored in a fixed oversized dtype."""
    rng = np.random.default_rng(7)
    sinograms = rng.normal(size=(2, 7, 300))

    result = detect_common_lines_all_pairs_blocked(
        sinograms,
        block_size=1,
    )

    assert np.issubdtype(result.index_1.dtype, np.unsignedinteger)
    assert np.iinfo(result.index_1.dtype).max >= 299
    assert result.index_1.dtype.itemsize <= 2


def test_main_input_errors_are_reported_clearly() -> None:
    """Keep validation tests limited to the two most important API failures."""
    with pytest.raises(ValueError, match="detector dimension"):
        detect_common_lines(
            np.zeros((4, 3)),
            np.zeros((5, 3)),
        )

    with pytest.raises(ValueError, match="block_size"):
        detect_common_lines_all_pairs_blocked(
            np.zeros((3, 4, 5)),
            block_size=0,
        )