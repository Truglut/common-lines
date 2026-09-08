import numpy as np
import pytest

from common_lines.centering import (
    estimate_reflection_shift,
    estimate_reflection_shifts,
    precenter_image_reflection,
    shift_image,
    shift_images,
)


def _gaussian_particle(
    height: int,
    width: int,
    *,
    shift_y: float = 0.0,
    shift_x: float = 0.0,
) -> np.ndarray:
    """Generate a smooth centered particle with an analytically known center."""
    yy, xx = np.mgrid[:height, :width]
    center_y = height // 2 + shift_y
    center_x = width // 2 + shift_x

    return np.exp(-0.5 * (((yy - center_y) / 8.0) ** 2 + ((xx - center_x) / 6.0) ** 2))


def _asymmetric_particle(
    height: int,
    width: int,
    *,
    shift_y: float = 0.0,
    shift_x: float = 0.0,
) -> np.ndarray:
    """Generate a smooth particle with no reflection symmetry."""
    yy, xx = np.mgrid[:height, :width]
    center_y = height // 2 + shift_y
    center_x = width // 2 + shift_x

    components = [
        (1.00, -7.0, -8.0, 3.0, 5.0),
        (0.75, 5.0, -1.0, 5.0, 2.5),
        (0.45, 1.5, 8.0, 2.0, 4.0),
    ]

    image = np.zeros((height, width), dtype=np.float64)
    for amplitude, dy, dx, sigma_y, sigma_x in components:
        image += amplitude * np.exp(
            -0.5
            * (
                ((yy - (center_y + dy)) / sigma_y) ** 2
                + ((xx - (center_x + dx)) / sigma_x) ** 2
            )
        )

    return image


def _axis_reflection_integer_shift(image: np.ndarray, axis: int) -> float:
    """Reference the best circular reflection lag without an orthogonal shift."""
    centered = image - image.mean()
    reflected = np.flip(centered, axis=axis)
    size = image.shape[axis]

    indices = np.arange(size)
    lags = ((indices + size // 2) % size) - size // 2
    scores = np.array(
        [np.sum(centered * np.roll(reflected, int(lag), axis=axis)) for lag in lags]
    )

    best_lag = lags[np.argmax(scores)]
    pixel_center_offset = size // 2 - (size - 1) / 2.0
    return -0.5 * best_lag + pixel_center_offset


@pytest.mark.parametrize("shape", [(65, 67), (64, 66)])
def test_centered_particle_has_nearly_zero_shift(shape):
    # The centering target is the same pixel used as the rotation center by
    # the common-line Radon transform.
    image = _gaussian_particle(*shape)

    shift_y, shift_x = estimate_reflection_shift(image)

    assert abs(shift_y) < 0.1
    assert abs(shift_x) < 0.1


@pytest.mark.parametrize(
    "particle_shift",
    [
        (0.0, 4.0),
        (-3.0, 0.0),
        (2.0, -5.0),
        (1.5, -2.5),
        (-3.25, 4.2),
    ],
)
def test_estimated_shift_is_opposite_to_particle_displacement(particle_shift):
    # Reflection doubles the displacement. The public convention returns the
    # correction to apply, so it must have the opposite sign.
    image = _gaussian_particle(
        65,
        67,
        shift_y=particle_shift[0],
        shift_x=particle_shift[1],
    )

    estimated = np.array(estimate_reflection_shift(image))
    expected = -np.array(particle_shift)

    np.testing.assert_allclose(estimated, expected, atol=0.25, rtol=0.0)


def test_even_sized_particle_uses_radon_center_pixel():
    # The Radon implementation rotates an even-sized image around size // 2.
    # Pre-centering must therefore leave an impulse at that pixel unchanged.
    image = np.zeros((64, 66), dtype=np.float64)
    image[32, 33] = 1.0

    estimated = np.array(estimate_reflection_shift(image))
    centered = precenter_image_reflection(image)

    np.testing.assert_array_equal(estimated, np.zeros(2))
    np.testing.assert_array_equal(centered, image)


def test_asymmetric_particle_translation_changes_estimate_by_opposite_shift():
    # An asymmetric particle may have an intrinsic reflection-centering bias.
    # Translating the whole particle should nevertheless change the estimate by
    # the opposite translation.
    particle_shift = np.array([3.0, -4.0])
    reference = _asymmetric_particle(65, 67)
    translated = _asymmetric_particle(
        65,
        67,
        shift_y=particle_shift[0],
        shift_x=particle_shift[1],
    )

    reference_estimate = np.array(estimate_reflection_shift(reference))
    translated_estimate = np.array(estimate_reflection_shift(translated))

    np.testing.assert_allclose(
        translated_estimate - reference_estimate,
        -particle_shift,
        atol=0.05,
        rtol=0.0,
    )


def test_asymmetric_particle_uses_only_axis_aligned_reflection_lags():
    # This configuration has a different optimum if the reflected image is
    # allowed to slide along the orthogonal direction. The estimator should
    # agree with independent 1D circular correlations along each axis instead.
    height = width = 33
    yy, xx = np.mgrid[:height, :width]
    center = height // 2

    components = [
        (0.90, 7.0, -9.0, 1.6, 2.8),
        (0.65, -2.5, 6.3, 1.7, 1.5),
        (1.40, -7.5, 6.5, 1.7, 1.4),
    ]

    image = np.zeros((height, width), dtype=np.float64)
    for amplitude, dy, dx, sigma_y, sigma_x in components:
        image += amplitude * np.exp(
            -0.5
            * (
                ((yy - (center + dy)) / sigma_y) ** 2
                + ((xx - (center + dx)) / sigma_x) ** 2
            )
        )

    estimated = np.array(estimate_reflection_shift(image))
    expected_integer_peak = np.array(
        [
            _axis_reflection_integer_shift(image, axis=-2),
            _axis_reflection_integer_shift(image, axis=-1),
        ]
    )

    # Subpixel parabolic refinement can move the estimate by at most half a
    # pixel from the discrete peak used by the independent reference.
    np.testing.assert_allclose(
        estimated,
        expected_integer_peak,
        atol=0.5,
        rtol=0.0,
    )


def test_precentering_is_approximately_idempotent():
    # Running the pre-centering stage again should make only a negligible
    # additional correction once the first pass has centered the particle.
    image = _gaussian_particle(65, 67, shift_y=-2.7, shift_x=3.6)

    once = precenter_image_reflection(image)
    twice = precenter_image_reflection(once)

    np.testing.assert_allclose(twice, once, atol=1.0e-2, rtol=0.0)


def test_shift_image_uses_y_x_order_and_does_not_wrap():
    # A single bright pixel makes the axis convention and border treatment
    # unambiguous without relying on the centering estimator.
    image = np.zeros((9, 11), dtype=np.float64)
    image[4, 9] = 1.0

    shifted = shift_image(image, shift_y=-2.0, shift_x=3.0)

    # The requested x shift would move the pixel beyond the right edge. It must
    # disappear rather than wrap around to the left side.
    assert np.max(np.abs(shifted)) == 0.0


def test_shift_images_applies_independent_batch_shifts():
    images = np.zeros((2, 9, 11), dtype=np.float64)
    images[:, 4, 5] = 1.0
    shifts = np.array([[2.0, -3.0], [-1.0, 2.0]])

    shifted = shift_images(images, shifts)

    assert shifted[0, 6, 2] == pytest.approx(1.0)
    assert shifted[1, 3, 7] == pytest.approx(1.0)
    assert shifted.sum() == pytest.approx(2.0)


def test_estimation_is_reasonably_robust_to_noise():
    # This is intentionally a loose test: pre-centering only needs to reduce a
    # moderate initial offset, not estimate it with registration-level accuracy.
    rng = np.random.default_rng(1234)
    particle_shift = np.array([2.5, -3.5])
    image = _gaussian_particle(65, 67, shift_y=2.5, shift_x=-3.5)
    image = image + rng.normal(scale=0.08, size=image.shape)

    estimated = np.array(estimate_reflection_shift(image))

    np.testing.assert_allclose(
        estimated,
        -particle_shift,
        atol=0.5,
        rtol=0.0,
    )


def test_constant_image_has_zero_estimated_shift():
    # A constant image contains no centering information. Returning zero is a
    # stable and harmless default for this degenerate case.
    images = np.ones((3, 32, 34), dtype=np.float32)

    estimated = estimate_reflection_shifts(images)

    np.testing.assert_array_equal(estimated, np.zeros((3, 2)))


@pytest.mark.parametrize("shape", [(65, 67), (64, 66)])
def test_batched_shift_estimation_matches_per_image(shape):
    n_images = 8
    images = np.random.normal(0, 1, size=(n_images, *shape))

    per_image = np.zeros(shape=(n_images, 2))
    for i in range(n_images):
        shift_y, shift_x = estimate_reflection_shift(images[i])

        per_image[i, 0] = shift_y
        per_image[i, 1] = shift_x

    batched = estimate_reflection_shifts(images)

    np.testing.assert_almost_equal(per_image, batched)


def test_small_precentering_case():
    image = np.zeros((5, 5), dtype=np.float64)
    center = np.array([2, 2])
    expected_shift = np.array([1, 1])

    image[tuple(center - expected_shift)] = 1.0

    expected_center = np.zeros((5, 5), dtype=np.float64)
    expected_center[tuple(center)] = 1.0

    estimated_shift = np.array(estimate_reflection_shift(image))
    centered_image = precenter_image_reflection(image)

    np.testing.assert_array_equal(estimated_shift, expected_shift)
    np.testing.assert_array_equal(centered_image, expected_center)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="image must have shape"):
        estimate_reflection_shift(np.zeros((2, 3, 4)))

    with pytest.raises(ValueError, match="image must have shape"):
        precenter_image_reflection(np.zeros((2, 32, 32)))

    with pytest.raises(ValueError, match="shifts must have shape"):
        shift_images(np.zeros((2, 16, 16)), np.zeros((2, 3)))

    bad = np.zeros((1, 16, 16))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimate_reflection_shifts(bad)
