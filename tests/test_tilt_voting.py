"""Consensus voting on the physical tilt interval, in radians."""

import numpy as np
import pytest

from common_lines import graph_estimation
from common_lines.euler import EulerAngles, euler_zyz_to_matrix
from common_lines.reconstitution.euler import vote_tilt
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


def test_graph_defaults_to_voting_with_a_five_degree_bandwidth() -> None:
    tilts = np.deg2rad([58.0, 59.0, 60.0, 61.0, 62.0, 120.0, 150.0])
    n_images = tilts.size + 2
    first = np.zeros((n_images, n_images))
    second = np.zeros_like(first)
    first[0, 2:] = np.pi / 2
    first[1, 2:] = np.pi / 2
    second[1, 2:] = tilts
    candidates, fallback = graph_estimation.estimate_relative_orientation_candidates(
        first, second, n_third=tilts.size, random_seed=4
    )
    assert not fallback[0, 1]
    expected = euler_zyz_to_matrix(EulerAngles(-np.pi / 2, np.pi / 3, np.pi / 2))
    np.testing.assert_allclose(candidates[0, 1, 0], expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("bins", [0, 1, 2.5])
def test_invalid_grid_is_rejected(bins: int) -> None:
    with pytest.raises(ValueError, match="bins"):
        vote_tilt(_triplets_with_tilts(1.0), bins=bins, sigma=0.1)


@pytest.mark.parametrize("sigma", [0.0, -0.1, np.nan, np.inf])
def test_invalid_bandwidth_is_rejected(sigma: float) -> None:
    with pytest.raises(ValueError, match="sigma"):
        vote_tilt(_triplets_with_tilts(1.0), bins=181, sigma=sigma)


def test_empty_and_invalid_geometry_are_rejected() -> None:
    with pytest.raises(ValueError, match="At least one"):
        vote_tilt(_triplets_with_tilts(np.empty((2, 0))), bins=181, sigma=0.1)
    triplet = CommonLineTriplet(
        CommonLineAngles(0.0, 0.0),
        CommonLineAngles(np.array([np.pi / 2, 0.0]), 0.0),
        CommonLineAngles(np.pi / 2, 1.0),
    )
    with np.errstate(all="raise"), pytest.raises(ValueError, match="geometry"):
        vote_tilt(triplet, bins=181, sigma=0.1)
