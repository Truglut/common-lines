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
        -(
            (columns - (center + offset_x)) ** 2
            + (rows - (center + offset_y)) ** 2
        )
        / (2.0 * 2.0**2)
    )

    theta = np.array([0.0, 30.0, 90.0, 135.0])
    sinogram = batch_common_line_sinograms(image, theta=theta, circle=True)

    detector_position = np.arange(size, dtype=np.float64) - center
    measured_centroid = (
        (detector_position[:, None] * sinogram).sum(axis=0)
        / sinogram.sum(axis=0)
    )
    theta_rad = np.deg2rad(theta)
    expected_centroid = (
        offset_x * np.cos(theta_rad) + offset_y * np.sin(theta_rad)
    )

    assert np.allclose(measured_centroid, expected_centroid, atol=2.0e-3)


def test_reversed_profile_is_represented_by_adding_180_degrees():
    """A reversed detector profile changes the orientation, not the sampled line."""
    profile = np.array([-2.0, 0.0, 4.0, 1.0, -1.0])
    sinogram_1 = profile[:, None]
    sinogram_2 = profile[::-1, None]

    result = detect_common_lines(
        sinogram_1,
        sinogram_2,
        theta_1=np.array([25.0]),
        theta_2=np.array([70.0]),
        allow_reversed=True,
    )

    assert bool(result.reversed_match[0])
    assert result.angle_1[0] == pytest.approx(25.0)
    assert result.angle_2[0] == pytest.approx(250.0)
    assert result.correlation[0] == pytest.approx(1.0)


def test_all_pairs_detection_orients_reversed_second_profiles():
    """The all-pairs detector applies the same oriented-angle convention."""
    profile = np.array([-2.0, 0.0, 4.0, 1.0, -1.0])
    sinograms = np.stack([profile[:, None], profile[::-1, None]])

    result = detect_common_lines_all_pairs(
        sinograms,
        theta=np.array([40.0]),
        allow_reversed=True,
    )

    assert bool(result.reversed_match[0, 1])
    assert result.angle_1[0, 1] == pytest.approx(40.0)
    assert result.angle_2[0, 1] == pytest.approx(220.0)
    assert result.correlation[0, 1] == pytest.approx(1.0)