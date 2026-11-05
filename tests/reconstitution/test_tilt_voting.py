"""Consensus voting on the physical tilt interval, in radians."""

import numpy as np
import pytest

from common_lines.reconstitution.euler import _weighted_histogram_vote, vote_tilt
from common_lines.reconstitution.triplets import CommonLineAngles, CommonLineTriplet


def _triplets_with_tilts(tilts: float | np.ndarray) -> CommonLineTriplet:
    # Orthogonal separations in the first two images give cos(tilt) = cos(c).
    return CommonLineTriplet(
        CommonLineAngles(0.0, 0.0),
        CommonLineAngles(np.pi / 2, 0.0),
        CommonLineAngles(np.pi / 2, tilts),
    )


@pytest.mark.parametrize("tilt", [0.0, 0.003, 0.73456, np.pi - 0.003, np.pi])
@pytest.mark.parametrize("shape", [(), (1,), (2, 3)])
def test_exact_consensus_is_not_quantized(tilt: float, shape: tuple[int, ...]) -> None:
    triplet = _triplets_with_tilts(np.full(shape, tilt))
    actual = vote_tilt(triplet, bins=181, sigma=np.deg2rad(5.0))
    assert actual == pytest.approx(tilt, abs=1e-12)


def test_scalar_fields_are_broadcast_before_voting() -> None:
    assert vote_tilt(
        _triplets_with_tilts(0.73456), bins=181, sigma=np.deg2rad(5.0)
    ) == pytest.approx(0.73456, abs=1e-12)


def test_dominant_cluster_wins_over_separated_outliers() -> None:
    tilts = np.deg2rad([58.0, 59.0, 60.0, 61.0, 62.0, 120.0, 150.0])
    actual = vote_tilt(_triplets_with_tilts(tilts), bins=181, sigma=np.deg2rad(5.0))
    assert actual == pytest.approx(np.deg2rad(60.0), abs=1e-12)
    assert abs(actual - np.mean(tilts)) > np.deg2rad(10.0)


def test_opposite_endpoints_do_not_vote_together() -> None:
    # Wrapping with period pi would combine the two smaller endpoint clusters.
    tilts = np.deg2rad([1.0, 2.0, 178.0, 179.0, 89.0, 90.0, 91.0])
    actual = vote_tilt(_triplets_with_tilts(tilts), bins=181, sigma=np.deg2rad(5.0))
    assert actual == pytest.approx(np.pi / 2, abs=1e-12)


def test_empty_and_invalid_geometry_are_rejected() -> None:
    with pytest.raises(ValueError):
        vote_tilt(_triplets_with_tilts(np.empty((2, 0))), bins=181, sigma=0.1)
    triplet = CommonLineTriplet(
        CommonLineAngles(0.0, 0.0),
        CommonLineAngles(np.array([np.pi / 2, 0.0]), 0.0),
        CommonLineAngles(np.pi / 2, 1.0),
    )
    with np.errstate(all="raise"), pytest.raises(ValueError):
        vote_tilt(triplet, bins=181, sigma=0.1)


def test_uniform_weights_reproduce_unweighted_voting() -> None:
    tilts = np.deg2rad([58.0, 59.0, 60.0, 61.0, 62.0, 120.0, 150.0])
    expected = vote_tilt(_triplets_with_tilts(tilts), bins=181, sigma=0.1)
    implicit = _weighted_histogram_vote(tilts, bins=181, sigma=0.1)
    explicit = _weighted_histogram_vote(
        tilts, bins=181, sigma=0.1, weights=np.ones_like(tilts)
    )
    assert implicit[0] == pytest.approx(expected, abs=1e-12)
    assert explicit == pytest.approx(implicit, abs=1e-12)


def test_weighted_cluster_and_weight_invariances() -> None:
    # The smaller cluster wins by total weight, not by witness count.
    tilts = np.array([0.5, 0.5, 0.5, 2.0, 2.0])
    weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    original_tilts, original_weights = tilts.copy(), weights.copy()
    expected = (2.0, 4.0 / 7.0)
    for scale in (1.0, 0.01, 100.0):
        scaled_weights = weights if scale == 1.0 else scale * weights
        result = _weighted_histogram_vote(
            tilts, bins=181, sigma=0.05, weights=scaled_weights
        )
        assert result == pytest.approx(expected, abs=1e-12)
    np.testing.assert_array_equal(tilts, original_tilts)
    np.testing.assert_array_equal(weights, original_weights)

    # A coarse grid puts the initial peak far from the only positive vote.
    # A zero-weight witness must not change the exponent shift in refinement.
    baseline = _weighted_histogram_vote(np.array([np.pi / 2]), bins=2, sigma=0.01)
    with np.errstate(divide="raise", invalid="raise"):
        augmented = _weighted_histogram_vote(
            np.array([0.0, np.pi / 2]),
            bins=2,
            sigma=0.01,
            weights=np.array([0.0, 1.0]),
        )
    assert baseline == pytest.approx((np.pi / 2, 1.0), abs=1e-12)
    assert augmented == pytest.approx(baseline, abs=1e-12)


def test_confidence_measures_absolute_gaussian_agreement() -> None:
    for tilts, weights in (
        (np.array([0.73456]), np.array([2.0])),
        (np.full(3, 0.73456), np.array([1.0, 2.0, 3.0])),
    ):
        assert _weighted_histogram_vote(
            tilts, bins=181, sigma=0.2, weights=weights
        ) == pytest.approx((0.73456, 1.0), abs=1e-12)

    result = _weighted_histogram_vote(np.array([0.9, 1.1]), bins=181, sigma=0.2)
    assert result == pytest.approx((1.0, np.exp(-0.125)), abs=1e-12)

    # At tilt 1, the weighted derivatives from these two votes cancel:
    # -0.2 * exp(-0.5) + 0.1 * w * exp(-0.125) = 0.
    # Their separation is below 2*sigma, so the Gaussian sum is unimodal.
    weights = np.array([1.0, 2.0 * np.exp(-0.375)])
    result = _weighted_histogram_vote(
        np.array([0.8, 1.1]), bins=181, sigma=0.2, weights=weights
    )
    expected_confidence = 3.0 * np.exp(-0.5) / weights.sum()
    assert result == pytest.approx((1.0, expected_confidence), abs=1e-12)


def test_voting_without_positive_support_is_undefined() -> None:
    for tilts, weights in (
        (np.array([]), None),
        (np.array([0.9, 1.1]), np.zeros(2)),
    ):
        with pytest.raises(ValueError):
            _weighted_histogram_vote(tilts, bins=181, sigma=0.1, weights=weights)
