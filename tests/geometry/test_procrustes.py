import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from common_lines.geometry.procrustes import nearest_rotation


def assert_is_rotation(matrix: np.ndarray, atol: float = 1e-12) -> None:
    """Check that a matrix belongs to SO(3)."""
    assert_allclose(matrix.T @ matrix, np.eye(3), atol=atol)
    assert_allclose(np.linalg.det(matrix), 1.0, atol=atol)


def test_preserves_an_exact_rotation() -> None:
    """An exact rotation should remain unchanged after projection."""
    expected = Rotation.from_euler(
        "xyz",
        [20.0, -35.0, 70.0],
        degrees=True,
    ).as_matrix()

    result = nearest_rotation(expected)

    assert_allclose(result, expected, atol=1e-12)
    assert_is_rotation(result)


def test_returns_a_rotation_for_a_noisy_matrix() -> None:
    """A perturbed rotation should be projected back onto SO(3)."""
    rng = np.random.default_rng(1234)

    original = Rotation.from_rotvec([0.3, -0.5, 0.8]).as_matrix()
    noisy = original + 0.05 * rng.standard_normal((3, 3))

    result = nearest_rotation(noisy)

    assert_is_rotation(result)
    assert np.linalg.norm(result - original, ord="fro") < 0.2


def test_corrects_a_reflection() -> None:
    """The result must have determinant +1 even for reflection-like inputs."""
    reflection = np.diag([1.0, 1.0, -1.0])

    result = nearest_rotation(reflection)

    assert_is_rotation(result)

    # The distance from a reflection to SO(3) is 2 in Frobenius norm.
    assert_allclose(
        np.linalg.norm(reflection - result, ord="fro"),
        2.0,
        atol=1e-12,
    )


def test_is_no_farther_than_sampled_rotations() -> None:
    """
    The projected matrix should be at least as close as randomly sampled
    rotations.

    This is not a proof of optimality, but helps detect mistakes in the SVD
    ordering or determinant correction.
    """
    rng = np.random.default_rng(5678)
    matrix = rng.standard_normal((3, 3))

    result = nearest_rotation(matrix)
    result_distance = np.linalg.norm(matrix - result, ord="fro")

    candidates = Rotation.random(1000, random_state=rng).as_matrix()
    candidate_distances = np.linalg.norm(
        candidates - matrix,
        axis=(1, 2),
    )

    assert result_distance <= candidate_distances.min() + 1e-12


@pytest.mark.parametrize(
    "invalid_matrix",
    [
        np.zeros((2, 2)),
        np.zeros((3, 4)),
        np.zeros(9),
    ],
)
def test_rejects_invalid_shapes(invalid_matrix: np.ndarray) -> None:
    """Only individual 3x3 matrices are accepted."""
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        nearest_rotation(invalid_matrix)


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_rejects_non_finite_values(invalid_value: float) -> None:
    """NaN and infinite values should produce a clear error."""
    matrix = np.eye(3)
    matrix[0, 0] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        nearest_rotation(matrix)


def test_rejects_complex_matrices() -> None:
    """The function is intended for real 3D rotation matrices."""
    matrix = np.eye(3, dtype=np.complex128)
    matrix[0, 0] += 1j

    with pytest.raises(ValueError, match="real-valued"):
        nearest_rotation(matrix)


def test_accepts_integer_input() -> None:
    """Integer arrays should be converted to floating point safely."""
    result = nearest_rotation(np.eye(3, dtype=int))

    assert np.issubdtype(result.dtype, np.floating)
    assert_allclose(result, np.eye(3))
    assert_is_rotation(result)
