import numpy as np

from common_lines.centering import (
    estimate_center_of_mass_shift,
    estimate_center_of_mass_shifts,
    estimate_shifts,
)
from common_lines.centering.center_of_mass import (
    compute_center_of_mass,
    compute_center_of_mass_batch,
)


def test_compute_center_of_mass_for_known_weighted_pixels():
    image = np.zeros((5, 7), dtype=np.float64)
    image[1, 2] = 1.0
    image[3, 4] = 3.0

    # for first coordinate (y), center of mass should be (1.0 * 1 + 3.0 * 3) / 4.0 = 10.0 / 4.0 = 2.5
    # for second coordinate (x), center of mass should be (1.0 * 2 + 3.0 * 4) / 4.0 = 14.0 / 4.0 = 3.5
    center = compute_center_of_mass(image)

    np.testing.assert_allclose(
        center,
        (2.5, 3.5),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_center_of_mass_shift_uses_radon_center_convention():
    image = np.zeros((6, 8), dtype=np.float64)
    # radon center is (6 // 2, 8 // 2) = (3, 4)
    image[2, 5] = 1.0
    # center of mass is (2, 5)

    # estimated shift should be (3, 4) - (2, 5) = (1, -1)
    shift = estimate_center_of_mass_shift(image)

    np.testing.assert_allclose(
        shift,
        (1.0, -1.0),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_compute_center_of_mass_batch_matches_scalar():
    images = np.zeros((3, 7, 9), dtype=np.float64)

    images[0, 2, 3] = 1.0

    images[1, 1, 2] = 1.0
    images[1, 5, 6] = 3.0

    images[2, 3, 4] = 2.0
    images[2, 4, 7] = 1.0

    batch_centers = compute_center_of_mass_batch(images)

    scalar_centers = np.array([compute_center_of_mass(image) for image in images])

    np.testing.assert_allclose(
        batch_centers,
        scalar_centers,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_estimate_center_of_mass_shifts_matches_scalar():
    images = np.zeros((3, 7, 9), dtype=np.float64)

    images[0, 2, 3] = 1.0

    images[1, 1, 2] = 1.0
    images[1, 5, 6] = 3.0

    images[2, 3, 4] = 2.0
    images[2, 4, 7] = 1.0

    batch_shifts = estimate_center_of_mass_shifts(images)

    scalar_shifts = np.array([estimate_center_of_mass_shift(image) for image in images])

    np.testing.assert_allclose(
        batch_shifts,
        scalar_shifts,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_centering_uses_supplied_images():
    from common_lines.preprocessing import (
        PreprocessingOptions,
        preprocess_images,
    )

    images = np.full((1, 9, 9), 5.0, dtype=np.float64)
    images[0, 2, 4] += 10.0

    # Background contributes to the center of mass when supplied unchanged.
    # Total mass is 81 * 5 + 10 = 415.
    np.testing.assert_allclose(
        estimate_shifts(images)[0],
        (20.0 / 415.0, 0.0),
        atol=1.0e-12,
        rtol=0.0,
    )

    # Explicit preprocessing removes the background, leaving one mass at (2, 4).
    options = PreprocessingOptions.default_from_images(images)
    processed = preprocess_images(images, options=options)

    np.testing.assert_allclose(
        estimate_shifts(processed)[0],
        (2.0, 0.0),
        atol=1.0e-12,
        rtol=0.0,
    )
