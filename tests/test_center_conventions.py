import pytest

import numpy as np

from common_lines.centering import precenter_images
from common_lines.radon import batch_common_line_sinograms


@pytest.mark.parametrize("size", [64, 65])
def test_precentering_and_radon_use_same_center(size):
    image = np.zeros((size, size), dtype=np.float64)
    image[size // 2 - 2, size // 2 + 3] = 1.0

    centered = precenter_images(image[None])[0]
    sinogram = batch_common_line_sinograms(
        centered,
        theta=np.array([0.0, 90.0]),
    )

    assert np.argmax(sinogram[:, 0]) == size // 2
    assert np.argmax(sinogram[:, 1]) == size // 2
