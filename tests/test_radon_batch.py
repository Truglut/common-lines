import time
import warnings

import numpy as np
import pytest
from skimage.transform import radon

from common_lines.radon import _batch_radon


def skimage_radon_loop(images, theta, *, circle=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return np.stack(
            [
                radon(
                    image,
                    theta=theta,
                    circle=circle,
                    preserve_range=True,
                )
                for image in images
            ],
            axis=0,
        )


def best_time(func, *args, repeat=3, **kwargs):
    best = float("inf")
    result = None

    for _ in range(repeat):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)

    return best, result


@pytest.mark.parametrize("circle", [True, False])
@pytest.mark.parametrize("shape", [(32, 32), (31, 31), (30, 40)])
def test_batch_radon_matches_skimage(circle, shape):
    rng = np.random.default_rng(0)

    batch_size = 5
    height, width = shape

    images = rng.normal(size=(batch_size, height, width)).astype(np.float64)
    theta = np.linspace(0, 180, 37, endpoint=False)

    expected = skimage_radon_loop(images, theta, circle=circle)
    actual = _batch_radon(images, theta=theta, circle=circle)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_single_image_matches_skimage_shape_and_values():
    rng = np.random.default_rng(1)

    image = rng.normal(size=(32, 32)).astype(np.float64)
    theta = np.linspace(0, 180, 45, endpoint=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = radon(
            image,
            theta=theta,
            circle=True,
            preserve_range=True,
        )
    actual = _batch_radon(image, theta=theta, circle=True)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.performance
def test_timing_against_skimage():
    rng = np.random.default_rng(2)

    batch_size = 128
    size = 64
    n_angles = 90

    images = rng.normal(size=(batch_size, size, size)).astype(np.float32)
    theta = np.linspace(0, 180, n_angles, endpoint=False)

    # Warm-up: first numba call includes compilation, so do not time it.
    _ = _batch_radon(images[:2], theta=theta[:5], circle=True)

    t_skimage, expected = best_time(
        skimage_radon_loop,
        images,
        theta,
        circle=True,
        repeat=3,
    )

    t_batch, actual = best_time(
        _batch_radon,
        images,
        theta,
        circle=True,
        repeat=3,
    )

    max_abs_error = np.max(np.abs(actual - expected))
    speedup = t_skimage / t_batch

    print()
    print(f"skimage loop time: {t_skimage:.4f} s")
    print(f"batch radon time:  {t_batch:.4f} s")
    print(f"speedup:           {speedup:.2f}x")
    print(f"max abs error:     {max_abs_error:.3e}")

    assert actual.shape == expected.shape
    assert max_abs_error < 5e-4

    # This assertion is intentionally mild because timing tests are machine-dependent.
    assert t_batch > 0.0
    assert t_skimage > 0.0
