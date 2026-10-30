"""Pair estimation: valid-witness sampling, aggregation, and fallback geometry."""

from typing import Literal

import numpy as np
import pytest

from common_lines.geometry.euler import EulerAngles, euler_zyz_to_matrix
from common_lines.graph_estimation import pairwise_orientation
from common_lines.reconstitution.euler import EulerPair
from common_lines.reconstitution.triplets import BoolArray, CommonLineTriplet


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
        pairwise_orientation.estimate_relative_orientation_candidates(
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


@pytest.mark.parametrize("method", ["averaging", "voting"])
def test_graph_filters_and_samples_reproducibly(
    monkeypatch: pytest.MonkeyPatch,
    method: Literal["averaging", "voting"],
) -> None:
    seed, n_third = 4, 2
    calls: dict[tuple[int, int], np.ndarray] = {}

    def valid_triplets(triplet: CommonLineTriplet) -> BoolArray:
        third = triplet.one_with_three.first_rad
        pair = triplet.one_with_two.first_rad
        if pair == 2:
            return np.asarray(third == 3)  # Fewer valid candidates than requested.
        return np.asarray((third != 2) & (pair != 0))

    def record_selection(triplet: CommonLineTriplet) -> np.ndarray:
        third = triplet.one_with_three.first_rad
        assert valid_triplets(triplet).all()
        assert 0 < len(third) <= n_third
        assert len(np.unique(third)) == len(third)
        i, j = int(triplet.one_with_two.second_rad), int(triplet.one_with_two.first_rad)
        assert (i, j) not in calls
        calls[i, j] = third.copy()
        return 0.1 * third

    def estimates(triplet: CommonLineTriplet) -> EulerPair:
        tilt = record_selection(triplet)
        zero = np.zeros_like(tilt)
        return EulerAngles(zero, tilt, zero), EulerAngles(zero, -tilt, zero)

    def vote(triplet: CommonLineTriplet, bins: int, sigma: float) -> float:
        return float(np.mean(record_selection(triplet)))

    monkeypatch.setattr(pairwise_orientation, "valid_euler_triplets", valid_triplets)
    monkeypatch.setattr(pairwise_orientation, "alignment_euler_angles", estimates)
    monkeypatch.setattr(pairwise_orientation, "vote_tilt", vote)
    first = np.broadcast_to(np.arange(6, dtype=float), (6, 6))
    second = first.T
    candidates, fallback = (
        pairwise_orientation.estimate_relative_orientation_candidates(
            first,
            second,
            n_third=n_third,
            random_seed=seed,
            method=method,
        )
    )
    selected = np.random.default_rng(seed).choice(
        [3, 4, 5], size=n_third, replace=False
    )
    np.testing.assert_array_equal(calls[0, 1], selected)
    np.testing.assert_array_equal(calls[0, 2], [3])
    assert not fallback[0, 1]
    assert fallback[1:, 0].all()
    assert all(j != 0 for i, j in calls)
    np.testing.assert_allclose(
        candidates[1, 0], pairwise_orientation._estimate_pairwise_fallback(0.0, 1.0)
    )
    calls.clear()
    repeated, repeated_fallback = (
        pairwise_orientation.estimate_relative_orientation_candidates(
            first,
            second,
            n_third=n_third,
            random_seed=seed,
            method=method,
        )
    )
    np.testing.assert_array_equal(repeated, candidates)
    np.testing.assert_array_equal(repeated_fallback, fallback)

    def broken_solver(triplet: CommonLineTriplet, **kwargs: object) -> EulerPair:
        raise ValueError("programming error")

    monkeypatch.setattr(
        pairwise_orientation,
        "alignment_euler_angles" if method == "averaging" else "vote_tilt",
        broken_solver,
    )
    with pytest.raises(ValueError, match="programming error"):
        pairwise_orientation.estimate_relative_orientation_candidates(
            first,
            second,
            n_third=n_third,
            random_seed=seed,
            method=method,
        )


def test_voting_uses_a_five_degree_bandwidth() -> None:
    tilts = np.deg2rad([58.0, 59.0, 60.0, 61.0, 62.0, 120.0, 150.0])
    n_images = tilts.size + 2
    first = np.zeros((n_images, n_images))
    second = np.zeros_like(first)
    first[0, 2:] = np.pi / 2
    first[1, 2:] = np.pi / 2
    second[1, 2:] = tilts
    candidates, fallback = (
        pairwise_orientation.estimate_relative_orientation_candidates(
            first, second, n_third=tilts.size, random_seed=4, method="voting"
        )
    )
    assert not fallback[0, 1]
    expected = euler_zyz_to_matrix(EulerAngles(-np.pi / 2, np.pi / 3, np.pi / 2))
    np.testing.assert_allclose(candidates[0, 1, 0], expected, rtol=0, atol=1e-12)
