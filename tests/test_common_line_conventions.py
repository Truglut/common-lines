import numpy as np
import pytest

from common_lines.detection import (
    detect_common_lines,
    detect_common_lines_all_pairs,
)
from common_lines.radon import batch_common_line_sinograms


def test_common_line_sinogram_angles_follow_image_coordinates():
    """Sinogram labels follow (cos(theta), sin(theta)) in column/row coordinates."""
    size = 65
    center = size // 2
    offset_x = 9.0
    offset_y = 6.0

    rows, columns = np.mgrid[:size, :size]
    image = np.exp(
        -((columns - (center + offset_x)) ** 2 + (rows - (center + offset_y)) ** 2)
        / (2.0 * 2.0**2)
    )

    theta = np.array([0.0, 30.0, 90.0, 135.0])
    sinogram = batch_common_line_sinograms(image, theta=theta, circle=True)

    detector_position = np.arange(size, dtype=np.float64) - center
    measured_centroid = (detector_position[:, None] * sinogram).sum(
        axis=0
    ) / sinogram.sum(axis=0)

    theta_rad = np.deg2rad(theta)
    expected_centroid = offset_x * np.cos(theta_rad) + offset_y * np.sin(theta_rad)

    assert np.allclose(measured_centroid, expected_centroid, atol=2.0e-3)


def test_reversed_profile_is_represented_by_adding_180_degrees():
    """
    Pairwise results store only indices and reconstruct oriented angles on demand.

    Reversing the detector profile does not change the sampled line index, but it
    changes the oriented direction in the second image by 180 degrees.
    """
    profile = np.array([-2.0, 0.0, 4.0, 1.0, -1.0])
    sinogram_1 = profile[:, None]
    sinogram_2 = profile[::-1, None]
    theta_1 = np.array([25.0])
    theta_2 = np.array([70.0])

    result = detect_common_lines(sinogram_1, sinogram_2, allow_reversed=True)
    angle_1, angle_2 = result.oriented_angles(theta_1, theta_2)

    assert result.reversed_match is not None
    assert bool(result.reversed_match[0])
    assert angle_1[0] == pytest.approx(25.0)
    assert angle_2[0] == pytest.approx(250.0)
    assert result.correlation[0] == pytest.approx(1.0)


def test_blocked_all_pairs_detection_uses_the_same_angle_convention():
    """
    The blocked all-pairs detector must apply the same reversed-profile convention
    as the non-blocked version.
    """
    profile = np.array([-2.0, 0.0, 4.0, 1.0, -1.0])
    sinograms = np.stack([profile[:, None], profile[::-1, None]])
    theta = np.array([40.0])

    graph = detect_common_lines_all_pairs(
        sinograms,
        allow_reversed=True,
        block_size=1,
    )
    angle_1, angle_2 = graph.oriented_angles(theta)

    assert graph.reversed_match is not None
    assert bool(graph.reversed_match[0, 1])
    assert angle_1[0, 1] == pytest.approx(40.0)
    assert angle_2[0, 1] == pytest.approx(220.0)
    assert graph.correlation[0, 1] == pytest.approx(1.0)
