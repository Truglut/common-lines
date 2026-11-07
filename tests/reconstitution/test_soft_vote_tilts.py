"""Geometry and reliability propagation in a full matrix voting update."""

import numpy as np
from scipy.spatial.transform import Rotation

from common_lines.geometry.forward import compute_intrinsic_common_line_angles
from common_lines.reconstitution.euler import (
    SoftVotingOptions,
    _voting_iteration,
    soft_vote_tilts,
)


def test_exact_geometry_recovers_pairwise_tilts() -> None:
    frames = Rotation.random(6, random_state=np.random.default_rng(12)).as_matrix()
    i, j = np.triu_indices(len(frames), k=1)
    angle_1, angle_2 = compute_intrinsic_common_line_angles(
        frames[:, None], frames[None, :]
    )
    # Directed pairs can choose opposite orientations of the same line.
    # The second array must be read explicitly, rather than using angle_1.T.
    assert not np.allclose(angle_2[i, j], angle_1.T[i, j])
    np.fill_diagonal(angle_1, np.nan)
    np.fill_diagonal(angle_2, np.nan)
    reliabilities = np.ones_like(angle_1)
    np.fill_diagonal(reliabilities, np.nan)
    for array in (angle_1, angle_2, reliabilities):
        array.flags.writeable = False

    # The third rows are plane normals; their dot product is the tilt cosine.
    expected = np.arccos(np.clip(frames[:, 2] @ frames[:, 2].T, -1.0, 1.0))
    np.fill_diagonal(expected, 0.0)
    tilts, confidence = _voting_iteration(
        angle_1, angle_2, reliabilities, voting_bins=181, voting_sigma=0.1, eps=1e-6
    )
    np.testing.assert_allclose(tilts, expected, rtol=0, atol=1e-11)
    np.testing.assert_allclose(confidence[i, j], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(confidence, confidence.T)
    assert np.isnan(np.diag(confidence)).all()

    result = soft_vote_tilts(
        angle_1, angle_2, options=SoftVotingOptions(voting_sigma=0.1, max_its=5)
    )
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-11)


def test_unequal_reliabilities_weight_witnesses_from_previous_update() -> None:
    # For pair (0, 1), orthogonal line separations make witnesses 2 and 3
    # yield tilts 0.8 and 1.1. Pair (2, 3) has a valid witness at image 0.
    angle_1 = np.array(
        [
            [np.nan, 0.0, np.pi / 2, np.pi / 2],
            [0.0, np.nan, np.pi / 2, np.pi / 2],
            [0.0, 0.8, np.nan, 0.7],
            [0.0, 1.1, 0.7, np.nan],
        ]
    )
    angle_2 = angle_1.T.copy()
    ratio = 2.0 * np.exp(-0.375)
    reliabilities = np.array(
        [
            [np.nan, 0.9, 0.5, 0.8],
            [0.9, np.nan, 0.4, 0.25 * ratio],
            [0.5, 0.4, np.nan, 0.7],
            [0.8, 0.25 * ratio, 0.7, np.nan],
        ]
    )
    original = reliabilities.copy()
    # Witness weights are 0.2 and 0.2*ratio. At tilt 1 their Gaussian
    # derivatives cancel, and the weighted support has this analytic value.
    expected_confidence = 3.0 * np.exp(-0.5) / (1.0 + ratio)
    # Some other pairs have parallel line separations in one witness.
    # They must be filtered before division, while valid witnesses still vote.
    with np.errstate(divide="raise", invalid="raise"):
        tilts, confidence = _voting_iteration(
            angle_1, angle_2, reliabilities, voting_bins=181, voting_sigma=0.2, eps=1e-6
        )
    np.testing.assert_allclose(tilts[0, 1], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        confidence[0, 1], expected_confidence, rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(reliabilities, original)
    np.testing.assert_array_equal(tilts, tilts.T)
    np.testing.assert_array_equal(confidence, confidence.T)

    # Relabeling images changes traversal order but must preserve the update.
    permutation = np.array([3, 2, 1, 0])
    index = np.ix_(permutation, permutation)
    with np.errstate(divide="raise", invalid="raise"):
        permuted_tilts, permuted_confidence = _voting_iteration(
            angle_1[index],
            angle_2[index],
            reliabilities[index],
            voting_bins=181,
            voting_sigma=0.2,
            eps=1e-6,
        )
    np.testing.assert_allclose(permuted_tilts, tilts[index], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        permuted_confidence, confidence[index], rtol=0, atol=1e-12
    )
