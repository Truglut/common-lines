"""Handedness reflection of frames and relative-orientation graphs."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common_lines.geometry.handedness import (
    mirror_relative_orientations_graph,
    mirror_transformation_matrices,
)


@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_mirror_frames_preserves_rotations_and_is_an_involution(
    batch_shape: tuple[int, ...],
) -> None:
    count = int(np.prod(batch_shape))
    rotations = (
        Rotation.random(count, random_state=12).as_matrix().reshape(*batch_shape, 3, 3)
    )
    original = rotations.copy()
    (mirrored,) = mirror_transformation_matrices((rotations,))

    # Conjugation by diag(1, 1, -1) changes only entries connecting z to x/y.
    expected = rotations.copy()
    expected[..., :2, 2] *= -1
    expected[..., 2, :2] *= -1
    np.testing.assert_array_equal(mirrored, expected)
    np.testing.assert_allclose(
        mirrored @ mirrored.swapaxes(-2, -1),
        np.broadcast_to(np.eye(3), mirrored.shape),
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(np.linalg.det(mirrored), 1.0, rtol=0, atol=1e-14)
    (restored,) = mirror_transformation_matrices((mirrored,))
    np.testing.assert_array_equal(restored, rotations)
    np.testing.assert_array_equal(rotations, original)
    assert not np.shares_memory(mirrored, rotations)


def test_mirror_frames_accepts_iterables_and_empty_input() -> None:
    rotations = Rotation.random(3, random_state=23).as_matrix()
    actual = mirror_transformation_matrices(rotation for rotation in rotations)
    assert isinstance(actual, tuple)
    assert len(actual) == 3
    (batched,) = mirror_transformation_matrices((rotations,))
    np.testing.assert_array_equal(np.stack(actual), batched)
    assert mirror_transformation_matrices(()) == ()


def test_mirrored_graph_matches_mirrored_frames() -> None:
    frames = Rotation.random(4, random_state=7).as_matrix()
    graph = frames[None, :] @ frames[:, None].swapaxes(-2, -1)
    original = graph.copy()
    mirrored = mirror_relative_orientations_graph(graph)
    mirrored_frames = np.stack(mirror_transformation_matrices(frames))
    expected = mirrored_frames[None, :] @ mirrored_frames[:, None].swapaxes(-2, -1)

    np.testing.assert_allclose(mirrored, expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(mirror_relative_orientations_graph(mirrored), graph)
    np.testing.assert_array_equal(graph, original)
    assert not np.shares_memory(mirrored, graph)
