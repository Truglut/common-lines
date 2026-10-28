import numpy as np
import pytest

from common_lines.alignment import (
    CommonLineAngles,
    CommonLineTriplet,
    compute_transformation_matrices,
    mirror_transformation_matrices,
)
from common_lines.detection import CommonLineResult, detect_common_lines
from common_lines.radon import batch_common_line_sinograms

THETA_DEG = np.arange(0.0, 180.0, 1.0)
FINE_THETA_DEG = np.arange(0.0, 180.0, 0.5)


# Expected frames for three simple orthogonal views. Rows 0 and 1 are the image
# axes in global coordinates; row 2 is the projection direction.
EXPECTED_R1 = np.eye(3)
EXPECTED_R2 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
EXPECTED_R3 = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def _first(value):
    """Extract the single value returned by one-pair common-line detection."""
    return np.asarray(value).reshape(-1)[0]


def _oriented_pair(result, theta_deg: np.ndarray) -> tuple[float, float]:
    """Convert one pairwise detection result into two scalar oriented angles."""
    angle_1, angle_2 = result.oriented_angles(theta_deg, theta_deg)
    return float(_first(angle_1)), float(_first(angle_2))


def _solve_detected_triplet(
    detected: tuple[object, object, object],
    theta_deg: np.ndarray,
):
    """Convert detected indices to angles and reconstruct both handednesses."""
    angle_12_in_1, angle_12_in_2 = _oriented_pair(detected[0], theta_deg)
    angle_13_in_1, angle_13_in_3 = _oriented_pair(detected[1], theta_deg)
    angle_23_in_2, angle_23_in_3 = _oriented_pair(detected[2], theta_deg)

    return compute_transformation_matrices(
        CommonLineTriplet(
            one_with_two=CommonLineAngles(
                first_rad=np.deg2rad(angle_12_in_1),
                second_rad=np.deg2rad(angle_12_in_2),
            ),
            one_with_three=CommonLineAngles(
                first_rad=np.deg2rad(angle_13_in_1),
                second_rad=np.deg2rad(angle_13_in_3),
            ),
            two_with_three=CommonLineAngles(
                first_rad=np.deg2rad(angle_23_in_2),
                second_rad=np.deg2rad(angle_23_in_3),
            ),
        )
    )


def _assert_rotation_matrix(R: np.ndarray, *, atol: float = 1.0e-12) -> None:
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=atol)
    assert np.isclose(np.linalg.det(R), 1.0, atol=atol)


def _rotation_distance_degrees(R_est: np.ndarray, R_true: np.ndarray) -> float:
    relative = R_est @ R_true.T
    cos_angle = (np.trace(relative) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def _global_rotation_distance_degrees(
    estimated: tuple[np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    """Return the largest rotation error across one complete triplet."""
    return max(
        _rotation_distance_degrees(R_est, R_expected)
        for R_est, R_expected in zip(estimated, expected)
    )


def _make_asymmetric_volume(size: int = 65) -> np.ndarray:
    """Small deterministic 3D object, stored as volume[z, y, x]."""
    grid = np.linspace(-1.0, 1.0, size)
    z, y, x = np.meshgrid(grid, grid, grid, indexing="ij")

    centers_zyx = np.array(
        [
            [-0.35, -0.20, -0.15],
            [0.28, 0.12, 0.22],
            [0.04, 0.38, -0.30],
            [0.10, -0.42, 0.35],
            [-0.25, 0.30, 0.15],
        ],
        dtype=np.float64,
    )
    sigmas = np.array([0.10, 0.14, 0.09, 0.12, 0.11])
    weights = np.array([1.00, 0.75, 0.90, 0.65, 0.55])

    volume = np.zeros_like(x)
    for (cz, cy, cx), sigma, weight in zip(centers_zyx, sigmas, weights):  # ty: ignore[not-iterable]
        r2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
        volume += weight * np.exp(-0.5 * r2 / sigma**2)

    return volume


def _axis_aligned_projections(volume: np.ndarray) -> np.ndarray:
    """
    Return three projections whose orientations are easy to inspect by hand.

    The returned image axes are:
      image 1: rows = y, cols = x, projected along z
      image 2: rows = z, cols = x, projected along y
      image 3: rows = z, cols = y, projected along x
    """
    image_1 = volume.sum(axis=0)  # y, x
    image_2 = volume.sum(axis=1)  # z, x
    image_3 = volume.sum(axis=2)  # z, y
    return np.stack([image_1, image_2, image_3])


def _detect_axis_aligned_common_lines(
    sinograms: np.ndarray,
) -> tuple[CommonLineResult, CommonLineResult, CommonLineResult]:
    result_12 = detect_common_lines(sinograms[0], sinograms[1], allow_reversed=True)
    result_13 = detect_common_lines(sinograms[0], sinograms[2], allow_reversed=True)
    result_23 = detect_common_lines(sinograms[1], sinograms[2], allow_reversed=True)
    return result_12, result_13, result_23


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues matrix for an active rotation around ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis

    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c

    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _make_arbitrary_frames() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct three non-orthogonal projection frames with known geometry.

    Each matrix follows the convention used by the orientation solver: rows 0 and
    1 are the image x/y axes in global coordinates, and row 2 is the projection
    direction. The first image is fixed to the identity.
    """
    deg = np.deg2rad

    R1 = np.eye(3)

    # Image 2 is tilted around the common x-axis with image 1.
    tilt_2 = deg(50.0)
    R2 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tilt_2), -np.sin(tilt_2)],
            [0.0, np.sin(tilt_2), np.cos(tilt_2)],
        ],
        dtype=np.float64,
    )

    # Image 3 shares a non-axis-aligned common line with image 1 and has an
    # additional in-plane rotation, so this is not an orthogonal special case.
    c13 = np.array([np.cos(deg(65.0)), np.sin(deg(65.0)), 0.0])
    n3 = _rotation_about_axis(c13, deg(58.0)) @ np.array([0.0, 0.0, 1.0])
    n3 = n3 / np.linalg.norm(n3)

    d13 = np.cross(n3, c13)
    d13 = d13 / np.linalg.norm(d13)

    in_plane = deg(35.0)
    r0 = np.cos(in_plane) * c13 - np.sin(in_plane) * d13
    r1 = np.sin(in_plane) * c13 + np.cos(in_plane) * d13

    R3 = np.vstack([r0, r1, n3])

    return R1, R2, R3


def _project_gaussian_volume_model(
    R: np.ndarray,
    *,
    size: int = 129,
    extent: float = 1.2,
) -> np.ndarray:
    """
    Project a continuous 3D Gaussian-mixture volume onto the frame ``R``.

    For isotropic 3D Gaussians, the projection is a 2D Gaussian whose centre is
    the 3D centre expressed in the two image axes. This keeps the projection step
    independent from the orientation-recovery code and avoids 3D interpolation.
    """
    centers_xyz = np.array(
        [
            [-0.15, -0.20, -0.35],
            [0.22, 0.12, 0.28],
            [-0.30, 0.38, 0.04],
            [0.35, -0.42, 0.10],
            [0.15, 0.30, -0.25],
        ],
        dtype=np.float64,
    )
    sigmas = np.array([0.08, 0.11, 0.07, 0.10, 0.09])
    weights = np.array([1.00, 0.75, 0.90, 0.65, 0.55])

    grid = np.linspace(-extent, extent, size)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")

    image = np.zeros((size, size), dtype=np.float64)
    for center, sigma, weight in zip(centers_xyz, sigmas, weights):
        center_x = center @ R[0]
        center_y = center @ R[1]
        r2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
        image += weight * np.exp(-0.5 * r2 / sigma**2)

    return image


def _detect_common_lines_for_three_images(
    sinograms: np.ndarray,
) -> tuple[CommonLineResult, CommonLineResult, CommonLineResult]:
    result_12 = detect_common_lines(sinograms[0], sinograms[1], allow_reversed=True)
    result_13 = detect_common_lines(sinograms[0], sinograms[2], allow_reversed=True)
    result_23 = detect_common_lines(sinograms[1], sinograms[2], allow_reversed=True)
    return result_12, result_13, result_23


def test_detect_common_lines_keeps_compact_array_outputs_for_a_single_pair():
    """
    Pairwise detection stores only compact index, score and reversal arrays.

    Oriented angles are reconstructed on demand from the angular grids instead
    of being retained in every result object.
    """
    projections = _axis_aligned_projections(_make_asymmetric_volume())
    sinograms = batch_common_line_sinograms(
        projections[:2],
        theta=THETA_DEG,
        circle=True,
    )

    result = detect_common_lines(sinograms[0], sinograms[1], allow_reversed=True)
    angle_1, angle_2 = result.oriented_angles(THETA_DEG, THETA_DEG)

    assert result.index_1.shape == (1,)
    assert result.index_2.shape == (1,)
    assert result.correlation.shape == (1,)
    assert result.reversed_match is not None
    assert result.reversed_match.shape == (1,)
    assert angle_1.shape == (1,)
    assert angle_2.shape == (1,)


def test_hand_checkable_orthogonal_common_lines_return_both_handednesses():
    """Exact common-line angles give the expected solution and its mirror."""
    direct, mirrored = compute_transformation_matrices(
        CommonLineTriplet(
            one_with_two=CommonLineAngles(first_rad=0.0, second_rad=0.0),
            one_with_three=CommonLineAngles(first_rad=np.pi / 2, second_rad=0.0),
            two_with_three=CommonLineAngles(first_rad=np.pi / 2, second_rad=np.pi / 2),
        )
    )
    expected = (EXPECTED_R1, EXPECTED_R2, EXPECTED_R3)
    expected_mirrored = mirror_transformation_matrices(expected)

    for rotations in (direct, mirrored):
        for R in rotations:
            _assert_rotation_matrix(R)

    for R_est, R_expected in zip(direct, expected):
        assert np.allclose(R_est, R_expected, atol=1.0e-12)

    for R_est, R_expected in zip(mirrored, expected_mirrored):
        assert np.allclose(R_est, R_expected, atol=1.0e-12)


def test_radon_detects_the_obvious_axis_aligned_common_lines():
    """The sinograms of the three axis-aligned projections have visible common lines."""
    projections = _axis_aligned_projections(_make_asymmetric_volume())
    sinograms = batch_common_line_sinograms(projections, theta=THETA_DEG, circle=True)

    result_12, result_13, result_23 = _detect_axis_aligned_common_lines(sinograms)

    angles_12 = _oriented_pair(result_12, THETA_DEG)
    angles_13 = _oriented_pair(result_13, THETA_DEG)
    angles_23 = _oriented_pair(result_23, THETA_DEG)

    assert angles_12 == pytest.approx((0.0, 0.0))
    assert angles_13 == pytest.approx((90.0, 0.0))
    assert angles_23 == pytest.approx((90.0, 90.0))

    assert _first(result_12.correlation) == pytest.approx(1.0)
    assert _first(result_13.correlation) == pytest.approx(1.0)
    assert _first(result_23.correlation) == pytest.approx(1.0)


def test_axis_aligned_volume_pipeline_recovers_one_global_handedness():
    """The complete pipeline recovers both globally consistent hands."""
    projections = _axis_aligned_projections(_make_asymmetric_volume())
    sinograms = batch_common_line_sinograms(projections, theta=THETA_DEG, circle=True)

    detected = _detect_axis_aligned_common_lines(sinograms)
    solutions = _solve_detected_triplet(detected, THETA_DEG)

    expected = (EXPECTED_R1, EXPECTED_R2, EXPECTED_R3)
    expected_mirrored = mirror_transformation_matrices(expected)

    for rotations in solutions:
        for R in rotations:
            _assert_rotation_matrix(R, atol=1.0e-10)

    errors_to_expected = [
        _global_rotation_distance_degrees(rotations, expected)
        for rotations in solutions
    ]
    errors_to_mirrored = [
        _global_rotation_distance_degrees(rotations, expected_mirrored)
        for rotations in solutions
    ]

    assert min(errors_to_expected) < 1.0e-6
    assert min(errors_to_mirrored) < 1.0e-6


def test_arbitrary_volume_pipeline_recovers_one_global_handedness():
    """A non-orthogonal case recovers a full triplet or its global mirror."""
    expected_rotations = _make_arbitrary_frames()
    expected_mirrored = mirror_transformation_matrices(expected_rotations)
    projections = np.stack(
        [_project_gaussian_volume_model(R) for R in expected_rotations]
    )

    sinograms = batch_common_line_sinograms(
        projections,
        theta=FINE_THETA_DEG,
        circle=True,
    )

    detected = _detect_common_lines_for_three_images(sinograms)
    solutions = _solve_detected_triplet(detected, FINE_THETA_DEG)

    for rotations in solutions:
        for R in rotations:
            _assert_rotation_matrix(R, atol=1.0e-8)

    errors_to_expected = [
        _global_rotation_distance_degrees(rotations, expected_rotations)
        for rotations in solutions
    ]
    errors_to_mirrored = [
        _global_rotation_distance_degrees(rotations, expected_mirrored)
        for rotations in solutions
    ]

    assert min(errors_to_expected) < 1.0
    assert min(errors_to_mirrored) < 1.0

    for result in detected:
        assert _first(result.correlation) > 0.999
