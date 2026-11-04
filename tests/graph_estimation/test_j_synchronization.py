"""Mathematical constraints behind sparse J-synchronization."""

import itertools

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common_lines.geometry.handedness import mirror_relative_orientations_graph
from common_lines.graph_estimation.j_synchronization import JSyncAdjacencyMatrix


def test_pair_indices_follow_upper_triangle_order() -> None:
    n_images = 6
    i, j = np.triu_indices(n_images, k=1)
    expected = np.arange(len(i))
    np.testing.assert_array_equal(
        JSyncAdjacencyMatrix.compute_pair_idx(i, j, n_images), expected
    )
    np.testing.assert_array_equal(
        JSyncAdjacencyMatrix.compute_pair_idx(j, i, n_images), expected
    )
    for index, (first, second) in enumerate(zip(i, j)):
        assert (
            JSyncAdjacencyMatrix.compute_pair_idx(int(first), int(second), n_images)
            == index
        )


@pytest.mark.parametrize("batch_size", [7, 512])
def test_adjacency_encodes_known_edge_partition_across_batches(batch_size: int) -> None:
    n_images = 6
    rng = np.random.default_rng(42)
    frames = Rotation.random(n_images, random_state=rng).as_matrix()
    truth = frames[None, :] @ frames[:, None].swapaxes(-2, -1)
    flips = rng.integers(0, 2, size=(n_images, n_images), dtype=bool)
    candidates = np.where(
        flips[..., None, None], mirror_relative_orientations_graph(truth), truth
    )

    pairs = list(itertools.combinations(range(n_images), 2))
    # Nodes are image pairs. Two distinct nodes share one triangle exactly when
    # they have an image in common; their sign says whether their hands agree.
    signs = np.array([-1.0 if flips[i, j] else 1.0 for i, j in pairs])
    neighbors = np.array(
        [
            [first != second and bool(set(first) & set(second)) for second in pairs]
            for first in pairs
        ]
    )
    expected = neighbors * signs[:, None] * signs[None, :]
    adjacency = JSyncAdjacencyMatrix.from_async_graph(candidates, batch_size=batch_size)
    np.testing.assert_array_equal(adjacency.matrix.toarray(), expected)

    eigenvector = adjacency.leading_eigenvector()
    # Eigenvectors have an arbitrary global sign; only their partition matters.
    recovered = np.where(eigenvector < 0, -1.0, 1.0)
    np.testing.assert_array_equal(recovered * recovered[0], signs * signs[0])
