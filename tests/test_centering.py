import numpy as np
import pytest

from common_lines.centering import (
    estimate_center_shift,
    estimate_center_shifts,
    precenter_image,
    precenter_images,
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
    center_y = (height - 1) / 2.0 + shift_y
    center_x = (width - 1) / 2.0 + shift_x

    return np.exp(-0.5 * (((yy - center_y) / 8.0) ** 2 + ((xx - center_x) / 6.0) ** 2))


@pytest.mark.parametrize("shape", [(65, 67), (64, 66)])
def test_centered_particle_has_nearly_zero_shift(shape):
    # Even and odd image sizes use slightly different pixel-center geometry.
    # A centered smooth particle should nevertheless need essentially no shift.
    image = _gaussian_particle(*shape)

    shift_y, shift_x = estimate_center_shift(image)

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

    estimated = np.array(estimate_center_shift(image))
    expected = -np.array(particle_shift)

    np.testing.assert_allclose(estimated, expected, atol=0.25, rtol=0.0)


def test_batch_estimation_handles_different_shifts_together():
    # This exercises the vectorized batch path rather than repeatedly calling
    # the scalar wrapper.
    particle_shifts = np.array(
        [
            [0.0, 0.0],
            [2.0, -4.0],
            [1.5, -2.5],
            [-3.25, 4.2],
        ]
    )
    images = np.stack(
        [
            _gaussian_particle(64, 66, shift_y=dy, shift_x=dx)
            for dy, dx in particle_shifts
        ]
    )

    estimated = estimate_center_shifts(images)

    assert estimated.shape == (len(images), 2)
    np.testing.assert_allclose(
        estimated,
        -particle_shifts,
        atol=0.25,
        rtol=0.0,
    )


def test_precentering_removes_most_of_the_residual_shift():
    # The centering method is deliberately approximate. The important property
    # is that a clearly displaced particle becomes very nearly centered.
    image = _gaussian_particle(65, 67, shift_y=3.2, shift_x=-4.4)

    centered = precenter_image(image)
    residual = np.array(estimate_center_shift(centered))

    assert np.linalg.norm(residual) < 0.25


def test_precentering_is_approximately_idempotent():
    # Running the pre-centering stage again should make only a negligible
    # additional correction once the first pass has centered the particle.
    image = _gaussian_particle(65, 67, shift_y=-2.7, shift_x=3.6)

    once = precenter_image(image)
    twice = precenter_image(once)

    np.testing.assert_allclose(twice, once, atol=1.0e-2, rtol=0.0)


def test_precenter_images_matches_scalar_precentering():
    images = np.stack(
        [
            _gaussian_particle(65, 67, shift_y=2.0, shift_x=-3.0),
            _gaussian_particle(65, 67, shift_y=-1.5, shift_x=4.0),
        ]
    )

    batch_result = precenter_images(images)
    scalar_result = np.stack([precenter_image(image) for image in images])

    np.testing.assert_allclose(batch_result, scalar_result, atol=1.0e-12, rtol=0.0)


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

    estimated = np.array(estimate_center_shift(image))

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

    estimated = estimate_center_shifts(images)

    np.testing.assert_array_equal(estimated, np.zeros((3, 2)))


@pytest.mark.parametrize("shape", [(65, 67), (64, 66)])
def test_batched_shift_estimation_matches_per_image(shape):
    n_images = 8
    images = np.random.normal(0, 1, size=(n_images, *shape))

    per_image = np.zeros(shape=(n_images, 2))
    for i in range(n_images):
        shift_y, shift_x = estimate_center_shift(images[i])

        per_image[i, 0] = shift_y
        per_image[i, 1] = shift_x

    batched = estimate_center_shifts(images)

    np.testing.assert_almost_equal(per_image, batched)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="image must have shape"):
        estimate_center_shift(np.zeros((2, 3, 4)))

    with pytest.raises(ValueError, match="images must have shape"):
        precenter_images(np.zeros((32, 32)))

    with pytest.raises(ValueError, match="shifts must have shape"):
        shift_images(np.zeros((2, 16, 16)), np.zeros((2, 3)))

    bad = np.zeros((1, 16, 16))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimate_center_shifts(bad)
