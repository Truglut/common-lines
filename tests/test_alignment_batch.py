"""Semantic regression coverage for batched geometry and graph aggregation."""

from typing import Literal

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common_lines import graph_estimation
from common_lines.euler import EulerAngles, euler_zyz_to_matrix
from common_lines.forward import compute_intrinsic_common_line_angles
from common_lines.reconstitution.euler import (
    EulerPair,
    alignment_euler_angles,
    valid_euler_triplets,
)
from common_lines.reconstitution.matrix import (
    compute_transformation_matrices,
    get_transformation_for_second_image,
)
from common_lines.reconstitution.triplets import (
    HANDEDNESS_REFLECTION,
    BoolArray,
    CommonLineAngles,
    CommonLineTriplet,
    FloatArray,
)


def _triplet(angles: list[float | np.ndarray]) -> CommonLineTriplet:
    return CommonLineTriplet(
        CommonLineAngles(*angles[:2]),
        CommonLineAngles(*angles[2:4]),
        CommonLineAngles(*angles[4:]),
    )


def _estimates(triplet: CommonLineTriplet) -> tuple[tuple[FloatArray, ...], ...]:
    """Express each solver's result as matrices for shared geometric assertions."""
    full = compute_transformation_matrices(triplet)
    second = get_transformation_for_second_image(triplet)
    euler = alignment_euler_angles(triplet)
    return (
        full[0] + full[1],
        second,
        tuple(euler_zyz_to_matrix(branch) for branch in euler),
    )


def test_broadcasting_preserves_scalar_geometry_and_inputs() -> None:
    frames = Rotation.random(9, random_state=np.random.default_rng(1)).as_matrix()
    # Fixed pair with five third images, a 2x3 grid of independent triplets,
    # and a length-one batch cover distinct broadcasting contracts.
    for first, second, third in (
        (frames[0], frames[1], frames[4:]),
        (frames[0], frames[1:3, None], frames[None, 4:7]),
        (frames[0], frames[1], frames[4:5]),
    ):
        angles = [
            *compute_intrinsic_common_line_angles(first, second),
            *compute_intrinsic_common_line_angles(first, third),
            *compute_intrinsic_common_line_angles(second, third),
        ]
        angles = [np.asarray(a) for a in angles]
        snapshots = [a.copy() for a in angles]
        for angle in angles:
            angle.flags.writeable = False
        broadcast = np.broadcast_arrays(*angles)
        shape = broadcast[0].shape
        triplet = _triplet(angles)
        valid = valid_euler_triplets(triplet)
        assert valid.shape == shape and valid.all()
        results = _estimates(triplet)
        for matrices in results:
            for matrix in matrices:
                assert matrix.shape == shape + (3, 3)
                assert matrix.dtype == np.float64
                np.testing.assert_allclose(
                    matrix @ matrix.swapaxes(-1, -2),
                    np.broadcast_to(np.eye(3), matrix.shape),
                    atol=1e-11,
                )
                np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1e-11)
            for direct, mirrored in zip(
                matrices[: len(matrices) // 2], matrices[len(matrices) // 2 :]
            ):
                np.testing.assert_allclose(
                    mirrored,
                    HANDEDNESS_REFLECTION @ direct @ HANDEDNESS_REFLECTION,
                    atol=1e-14,
                )
        for index in np.ndindex(shape):
            scalar_triplet = _triplet([float(a[index]) for a in broadcast])
            valid = valid_euler_triplets(scalar_triplet)
            assert valid.shape == () and valid
            scalar = _estimates(scalar_triplet)
            for actual, expected in zip(results, scalar):
                for matrix, reference in zip(actual, expected):
                    assert reference.shape == (3, 3)
                    np.testing.assert_allclose(matrix[index], reference, atol=1e-12)
        truth = second @ first.swapaxes(-1, -2)
        for matrices in results[1:]:
            errors = [np.max(np.abs(m - truth), axis=(-2, -1)) for m in matrices]
            assert np.max(np.minimum(*errors)) < 1e-11
        for angle, snapshot in zip(angles, snapshots):
            np.testing.assert_array_equal(angle, snapshot)


def test_invalid_geometry_is_filtered_before_solving() -> None:
    # One valid triplet, reference/second-image parallelism, and impossible geometry.
    angles = [
        0.0,
        0.0,
        np.array([np.pi / 2, 0.0, 1.0, 1.0]),
        0.0,
        np.array([np.pi / 2, 1.0, 0.0, 1.0]),
        np.array([np.pi / 2, 0.5, 0.5, 2.01]),
    ]
    with np.errstate(all="raise"):
        valid = valid_euler_triplets(_triplet(angles))
    np.testing.assert_array_equal(valid, [True, False, False, False])
    fields = np.broadcast_arrays(*angles)
    selected = _triplet([field[valid] for field in fields])
    assert all(
        np.isfinite(m).all() for matrices in _estimates(selected) for m in matrices
    )
    for solver in (
        compute_transformation_matrices,
        get_transformation_for_second_image,
        alignment_euler_angles,
    ):
        # One invalid member invalidates the call, rather than returning partial results.
        with np.errstate(all="raise"), pytest.raises(ValueError):
            solver(_triplet(angles))
        for index in range(1, len(valid)):
            invalid = _triplet([float(field[index]) for field in fields])
            with np.errstate(all="raise"), pytest.raises(ValueError):
                solver(invalid)


def test_empty_batches_and_incompatible_shapes() -> None:
    triplet = _triplet([0, 0, np.empty((2, 0)), 0, 1, 1])
    assert valid_euler_triplets(triplet).shape == (2, 0)
    for matrices in _estimates(triplet):
        assert all(m.shape == (2, 0, 3, 3) for m in matrices)
    for solver in (
        compute_transformation_matrices,
        get_transformation_for_second_image,
        alignment_euler_angles,
        valid_euler_triplets,
    ):
        with pytest.raises(ValueError):
            solver(_triplet([np.ones(2), 0, np.ones(3), 0, 1, 1]))


def test_solver_specific_tolerances_and_raw_estimates() -> None:
    triplet = _triplet([0, 0, np.pi / 2, 0, np.pi / 2, 0])
    # Degeneracy confined to reconstructing image 3 leaves image 2 recoverable.
    with pytest.raises(ValueError):
        compute_transformation_matrices(triplet)
    assert valid_euler_triplets(triplet)
    get_transformation_for_second_image(triplet)
    alignment_euler_angles(triplet)

    triplet = _triplet([0, 0, 0.2, 0, 0.2, 0.4 + 2e-7])
    raw, _ = get_transformation_for_second_image(triplet)
    assert not valid_euler_triplets(triplet)  # Cosine and squared-z tolerances differ.
    with pytest.raises(ValueError):
        alignment_euler_angles(triplet)
    with pytest.raises(ValueError):
        get_transformation_for_second_image(triplet, eps=1e-9)
    assert (
        np.max(np.abs(raw @ raw.T - np.eye(3))) > 1e-9
    )  # No implicit SO(3) projection.


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

    monkeypatch.setattr(graph_estimation, "valid_euler_triplets", valid_triplets)
    monkeypatch.setattr(graph_estimation, "alignment_euler_angles", estimates)
    monkeypatch.setattr(graph_estimation, "vote_tilt", vote)
    first = np.broadcast_to(np.arange(6, dtype=float), (6, 6))
    second = first.T
    candidates, fallback = graph_estimation.estimate_relative_orientation_candidates(
        first,
        second,
        n_third=n_third,
        random_seed=seed,
        method=method,
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
        candidates[1, 0], graph_estimation._estimate_pairwise_fallback(0.0, 1.0)
    )
    calls.clear()
    repeated, repeated_fallback = (
        graph_estimation.estimate_relative_orientation_candidates(
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
        graph_estimation,
        "alignment_euler_angles" if method == "averaging" else "vote_tilt",
        broken_solver,
    )
    with pytest.raises(ValueError, match="programming error"):
        graph_estimation.estimate_relative_orientation_candidates(
            first,
            second,
            n_third=n_third,
            random_seed=seed,
            method=method,
        )
