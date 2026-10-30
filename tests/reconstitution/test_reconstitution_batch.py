"""Broadcasting, validation, and numerical conventions of triplet solvers."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common_lines.geometry.euler import euler_zyz_to_matrix
from common_lines.geometry.forward import compute_intrinsic_common_line_angles
from common_lines.reconstitution.euler import (
    alignment_euler_angles,
    valid_euler_triplets,
)
from common_lines.reconstitution.matrix import (
    compute_transformation_matrices,
    get_transformation_for_second_image,
)
from common_lines.reconstitution.triplets import (
    HANDEDNESS_REFLECTION,
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
