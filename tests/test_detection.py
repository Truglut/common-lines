import numpy as np
import pytest

from common_lines.detection import (
    detect_common_lines,
    detect_common_lines_all_pairs,
)


def brute_force_common_line(s1, s2, *, eps=1.0e-8):
    """
    Reference implementation used in one test.

    This intentionally uses explicit loops over all angle pairs. It is slower
    than the vectorized implementation, but easier to trust for validation.
    """
    n_angles_1 = s1.shape[1]
    n_angles_2 = s2.shape[1]

    best_i = 0
    best_j = 0
    best_corr = -np.inf

    for i in range(n_angles_1):
        x = s1[:, i]
        x = x - x.mean()
        norm_x = np.linalg.norm(x)

        for j in range(n_angles_2):
            y = s2[:, j]
            y = y - y.mean()
            norm_y = np.linalg.norm(y)

            if norm_x <= eps or norm_y <= eps:
                corr = 0.0
            else:
                corr = np.dot(x, y) / (norm_x * norm_y)

            if corr > best_corr:
                best_corr = corr
                best_i = i
                best_j = j

    return best_i, best_j, best_corr


def test_detect_common_lines_known_pair():
    """
    Check that the function identifies a deliberately inserted common line.

    We create two random sinograms and then overwrite one column in each with
    the same profile, up to a positive affine transformation. Since Pearson
    correlation is invariant to positive scaling and shifts, the expected
    correlation is 1.
    """
    rng = np.random.default_rng(0)

    n_detectors = 32
    n_angles = 12

    s1 = rng.normal(size=(n_detectors, n_angles))
    s2 = rng.normal(size=(n_detectors, n_angles))

    common_profile = rng.normal(size=n_detectors)

    expected_i = 3
    expected_j = 8

    s1[:, expected_i] = common_profile
    s2[:, expected_j] = 2.5 * common_profile + 10.0

    result = detect_common_lines(s1, s2)

    assert result.index_1.shape == (1,)
    assert result.index_2.shape == (1,)
    assert result.correlation.shape == (1,)

    assert result.index_1[0] == expected_i
    assert result.index_2[0] == expected_j
    assert result.correlation[0] == pytest.approx(1.0)


def test_detect_common_lines_known_batch():
    """
    Check pairwise batch behaviour.

    Each element of the first batch is compared only with the corresponding
    element of the second batch. For every batch element, we insert a known
    matching profile at different angle indices.
    """
    rng = np.random.default_rng(1)

    batch_size = 5
    n_detectors = 32
    n_angles = 16

    s1 = rng.normal(size=(batch_size, n_detectors, n_angles))
    s2 = rng.normal(size=(batch_size, n_detectors, n_angles))

    expected_i = np.array([1, 4, 7, 10, 13])
    expected_j = np.array([3, 6, 9, 12, 15])

    for b in range(batch_size):
        common_profile = rng.normal(size=n_detectors)
        s1[b, :, expected_i[b]] = common_profile
        s2[b, :, expected_j[b]] = -3.0 + 4.0 * common_profile

    result = detect_common_lines(s1, s2)

    np.testing.assert_array_equal(result.index_1, expected_i)
    np.testing.assert_array_equal(result.index_2, expected_j)
    np.testing.assert_allclose(result.correlation, np.ones(batch_size), atol=1.0e-12)


def test_detect_common_lines_matches_brute_force():
    """
    Compare the vectorized implementation against a simple brute-force version.

    This does not rely on manually inserted common lines. Instead, it uses
    random sinograms and checks that both implementations select the same best
    angle pair and the same correlation value.
    """
    rng = np.random.default_rng(2)

    batch_size = 4
    n_detectors = 24
    n_angles_1 = 9
    n_angles_2 = 11

    s1 = rng.normal(size=(batch_size, n_detectors, n_angles_1))
    s2 = rng.normal(size=(batch_size, n_detectors, n_angles_2))

    result = detect_common_lines(s1, s2)

    expected_i = []
    expected_j = []
    expected_corr = []

    for b in range(batch_size):
        i, j, corr = brute_force_common_line(s1[b], s2[b])
        expected_i.append(i)
        expected_j.append(j)
        expected_corr.append(corr)

    np.testing.assert_array_equal(result.index_1, np.array(expected_i))
    np.testing.assert_array_equal(result.index_2, np.array(expected_j))
    np.testing.assert_allclose(result.correlation, np.array(expected_corr))


def test_detect_common_lines_returns_angles():
    """
    Check that selected column indices are converted back to angle values.

    The detection itself is still based on column indices. Here we provide a
    theta array and verify that the returned angles match the selected indices.
    """
    rng = np.random.default_rng(3)

    n_detectors = 32
    n_angles = 10

    theta = np.linspace(0, 180, n_angles, endpoint=False)

    s1 = rng.normal(size=(n_detectors, n_angles))
    s2 = rng.normal(size=(n_detectors, n_angles))

    common_profile = rng.normal(size=n_detectors)

    s1[:, 2] = common_profile
    s2[:, 7] = common_profile

    result = detect_common_lines(
        s1,
        s2,
        theta_1=theta,
        theta_2=theta,
    )

    assert result.index_1[0] == 2
    assert result.index_2[0] == 7
    assert result.angle_1[0] == pytest.approx(theta[2])
    assert result.angle_2[0] == pytest.approx(theta[7])


def test_detect_common_lines_can_return_correlation_matrix():
    """
    Check the optional full correlation matrix output.

    The function normally returns only the best match. With
    return_correlation_matrix=True, it should also return the full matrix of
    all angle-angle scores for each batch element.
    """
    rng = np.random.default_rng(4)

    s1 = rng.normal(size=(2, 20, 6))
    s2 = rng.normal(size=(2, 20, 7))

    result = detect_common_lines(
        s1,
        s2,
        return_correlation_matrix=True,
    )

    assert result.correlation_matrix is not None
    assert result.correlation_matrix.shape == (2, 6, 7)

    for b in range(2):
        i = result.index_1[b]
        j = result.index_2[b]
        assert result.correlation[b] == result.correlation_matrix[b, i, j]


def test_detect_common_lines_with_reversed_detector_axis():
    """
    Check matching when the common line appears with opposite orientation.

    We insert a profile in the first sinogram and the reversed profile in the
    second. The match should only reach correlation 1 when allow_reversed=True.
    """
    rng = np.random.default_rng(5)

    n_detectors = 32
    n_angles = 12

    s1 = rng.normal(size=(n_detectors, n_angles))
    s2 = rng.normal(size=(n_detectors, n_angles))

    common_profile = rng.normal(size=n_detectors)

    s1[:, 4] = common_profile
    s2[:, 9] = common_profile[::-1]

    result = detect_common_lines(
        s1,
        s2,
        allow_reversed=True,
    )

    assert result.index_1[0] == 4
    assert result.index_2[0] == 9
    assert result.reversed_match[0]
    assert result.correlation[0] == pytest.approx(1.0)


def test_detect_common_lines_handles_constant_columns():
    """
    Check numerical stability for constant sinogram columns.

    Constant columns have zero variance, so their Pearson correlation is
    undefined. In this implementation they are normalized to zero vectors,
    producing finite correlation values equal to zero.
    """
    s1 = np.ones((20, 5))
    s2 = np.ones((20, 5))

    result = detect_common_lines(s1, s2)

    assert np.isfinite(result.correlation).all()
    assert result.correlation[0] == pytest.approx(0.0)


def test_detect_common_lines_broadcasts_single_sinogram_against_batch():
    """
    Check broadcasting of a single sinogram against a batch.

    This allows calling detect_common_lines(single, batch) without manually
    repeating the single sinogram B times.
    """
    rng = np.random.default_rng(6)

    n_detectors = 24
    n_angles = 8
    batch_size = 3

    s1 = rng.normal(size=(n_detectors, n_angles))
    s2 = rng.normal(size=(batch_size, n_detectors, n_angles))

    result = detect_common_lines(s1, s2)

    assert result.index_1.shape == (batch_size,)
    assert result.index_2.shape == (batch_size,)
    assert result.correlation.shape == (batch_size,)


def test_detect_common_lines_all_pairs_known_cross_pair():
    """
    Check all-pairs detection on a known pair inside one batch.

    We insert a shared profile between sinogram 0 and sinogram 1. The all-pairs
    function should find that match in the [0, 1] entry of the output matrices.
    """
    rng = np.random.default_rng(7)

    batch_size = 3
    n_detectors = 32
    n_angles = 10

    sinograms = rng.normal(size=(batch_size, n_detectors, n_angles))

    common_profile = rng.normal(size=n_detectors)

    sinograms[0, :, 2] = common_profile
    sinograms[1, :, 6] = 5.0 * common_profile - 1.0

    result = detect_common_lines_all_pairs(sinograms)

    assert result.index_1.shape == (batch_size, batch_size)
    assert result.index_2.shape == (batch_size, batch_size)
    assert result.correlation.shape == (batch_size, batch_size)

    assert result.index_1[0, 1] == 2
    assert result.index_2[0, 1] == 6
    assert result.correlation[0, 1] == pytest.approx(1.0)


def test_detect_common_lines_all_pairs_matches_pairwise_function():
    """
    Check that all-pairs detection is consistent with pairwise detection.

    For every pair (i, j), the result from detect_common_lines_all_pairs should
    match calling detect_common_lines on sinograms[i] and sinograms[j].
    """
    rng = np.random.default_rng(8)

    batch_size = 4
    n_detectors = 20
    n_angles = 7

    sinograms = rng.normal(size=(batch_size, n_detectors, n_angles))

    all_pairs = detect_common_lines_all_pairs(sinograms)

    for i in range(batch_size):
        for j in range(batch_size):
            pair = detect_common_lines(sinograms[i], sinograms[j])

            assert all_pairs.index_1[i, j] == pair.index_1[0]
            assert all_pairs.index_2[i, j] == pair.index_2[0]
            assert all_pairs.correlation[i, j] == pytest.approx(pair.correlation[0])


def test_detect_common_lines_rejects_mismatched_detector_size():
    """
    Check that sinograms with different detector dimensions are rejected.

    Column correlations require both profiles to have the same length. Different
    numbers of angles are allowed, but different detector dimensions are not.
    """
    s1 = np.zeros((20, 8))
    s2 = np.zeros((21, 8))

    with pytest.raises(ValueError, match="same detector dimension"):
        detect_common_lines(s1, s2)


def test_detect_common_lines_rejects_incompatible_batch_sizes():
    """
    Check batch-size validation.

    The function supports equal batch sizes, or broadcasting when one batch has
    size 1. Other combinations are ambiguous and should raise an error.
    """
    s1 = np.zeros((2, 20, 8))
    s2 = np.zeros((3, 20, 8))

    with pytest.raises(ValueError, match="Batch dimensions must match"):
        detect_common_lines(s1, s2)


def test_detect_common_lines_rejects_invalid_theta_length():
    """
    Check that theta arrays must match the number of sinogram columns.

    This avoids returning angle values that do not correspond to the actual
    columns being searched.
    """
    s1 = np.zeros((20, 8))
    s2 = np.zeros((20, 8))
    theta = np.linspace(0, 180, 7, endpoint=False)

    with pytest.raises(ValueError, match="theta_1 must have length 8"):
        detect_common_lines(s1, s2, theta_1=theta)
