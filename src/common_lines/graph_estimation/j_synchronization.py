"""
J-synchronization method from
Pragier, G., Greenberg, I., Cheng, X., & Shkolnisky, Y. (2016).
A Graph Partitioning Approach to Simultaneous Angular Reconstitution.
IEEE transactions on computational imaging, 2(3), 323–334.
https://doi.org/10.1109/TCI.2016.2557076
Section II
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from common_lines.reconstitution.triplets import HANDEDNESS_REFLECTION


class JSyncAdjacencyMatrix:
    """Sparse adjacency matrix for joint J-synchronization of orientations over image pairs."""

    __slots__ = ("matrix", "n_images")

    n_images: int
    matrix: sp.csr_array

    def __init__(self, n_images: int, data: sp.csr_array | None):
        expected_length = math.comb(n_images, 2)

        if data is None:
            data = sp.csr_array((expected_length, expected_length))
        elif data.shape != (expected_length, expected_length):
            raise ValueError(
                f"Invalid matrix shape {data.shape}. "
                f"Expected ({expected_length}, {expected_length}) for n_images={n_images}."
            )

        self.n_images = n_images
        self.matrix = sp.csr_array(data, dtype=np.float64)

    @property
    def n_nodes(self) -> int:
        return self.matrix.shape[0]

    @property
    def nnz(self) -> int:
        return self.matrix.nnz

    @property
    def n_edges(self) -> int:
        return 3 * math.comb(self.n_images, 3)

    def leading_eigenvector(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        v0 = rng.standard_normal(self.n_nodes)

        _, eigenvectors = spla.eigsh(self.matrix, k=1, which="LA", v0=v0)

        return eigenvectors[:, 0]

    @staticmethod
    def compute_pair_idx(
        i: np.ndarray | int, j: np.ndarray | int, n_images: int
    ) -> np.ndarray | int:
        i, j = np.asarray(i), np.asarray(j)

        if (i == j).any():
            raise ValueError(
                "Input indices must be distinct. The synchronization graph does not "
                "have nodes for diagonal pairs."
            )

        first_idx = np.minimum(i, j)
        second_idx = np.maximum(i, j)
        return (
            first_idx * n_images
            - first_idx * (first_idx + 1) // 2
            + (second_idx - first_idx - 1)
        )

    def pair_idx(self, i: np.ndarray | int, j: np.ndarray | int) -> np.ndarray | int:
        return self.compute_pair_idx(i, j, self.n_images)

    @classmethod
    def from_async_graph(
        cls, async_graph: np.ndarray, batch_size: int = 512
    ) -> JSyncAdjacencyMatrix:
        if (
            async_graph.ndim != 4
            or async_graph.shape[-2:] != (3, 3)
            or async_graph.shape[0] != async_graph.shape[1]
        ):
            raise ValueError(
                "async_graph must be an array of shape (n_images, n_images, 3, 3)"
            )

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        n_images = async_graph.shape[0]

        n_triplets = math.comb(n_images, 3)
        data = np.empty(3 * n_triplets, dtype=np.float64)
        locations = np.empty((3 * n_triplets, 2), dtype=np.intp)

        triplets = itertools.combinations(range(n_images), 3)

        data_start = 0
        while True:
            batch_triplets = list(itertools.islice(triplets, batch_size))
            if not batch_triplets:
                break

            i, j, k = np.asarray(batch_triplets, dtype=np.intp).T

            ij_pairs = async_graph[i, j]
            jk_pairs = async_graph[j, k]
            ik_pairs = async_graph[i, k]

            ij_group, jk_group, ik_group = find_optimum_combination(
                ij_pairs, jk_pairs, ik_pairs
            )

            ij_idx = JSyncAdjacencyMatrix.compute_pair_idx(i, j, n_images)
            jk_idx = JSyncAdjacencyMatrix.compute_pair_idx(j, k, n_images)
            ik_idx = JSyncAdjacencyMatrix.compute_pair_idx(i, k, n_images)

            sign_ij_jk = np.where(ij_group == jk_group, 1.0, -1.0)
            sign_jk_ik = np.where(jk_group == ik_group, 1.0, -1.0)
            sign_ik_ij = np.where(ik_group == ij_group, 1.0, -1.0)

            batch_length = len(i)
            target = slice(data_start, data_start + 3 * batch_length)

            data[target] = np.stack(
                (sign_ij_jk, sign_jk_ik, sign_ik_ij), axis=1
            ).ravel()

            locations[target, 0] = np.stack((ij_idx, jk_idx, ik_idx), axis=1).ravel()
            locations[target, 1] = np.stack((jk_idx, ik_idx, ij_idx), axis=1).ravel()

            data_start += 3 * batch_length

        length = math.comb(n_images, 2)
        asym_matrix = sp.coo_array(
            (data, (locations[:, 0], locations[:, 1])),
            shape=(length, length),
            dtype=np.float64,
        ).tocsr()
        matrix = asym_matrix + asym_matrix.T

        return cls(n_images=n_images, data=matrix)


def find_optimum_combination(
    ij_pairs: np.ndarray, jk_pairs: np.ndarray, ik_pairs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (ij_pairs.shape == jk_pairs.shape == ik_pairs.shape):
        raise ValueError(
            "All rotation pairs must have the same shape, got shapes "
            f"{ij_pairs.shape}, {jk_pairs.shape}, {ik_pairs.shape}."
        )
    if ij_pairs.ndim < 2 or ij_pairs.shape[-2:] != (3, 3):
        raise ValueError(
            f"Input arrays must be of shape (..., 3, 3), got {ij_pairs.shape}."
        )

    J = HANDEDNESS_REFLECTION

    # We can fix one branch to an arbitrary value, since we only need to check
    # each edge for same or different group
    jk_choices = np.stack((jk_pairs, J @ jk_pairs @ J), axis=0)
    ik_choices = np.stack((ik_pairs, J @ ik_pairs @ J), axis=0)

    ij = ij_pairs[None, None, None, ...]
    jk = jk_choices[None, :, None, ...]
    ik = ik_choices[None, None, :, ...]

    residuals = ik.swapaxes(-2, -1) @ jk @ ij - np.eye(3)
    costs = np.linalg.norm(residuals, ord="fro", axis=(-2, -1))

    flat_groups = costs.reshape(-1, *costs.shape[3:]).argmin(axis=0)

    return np.unravel_index(flat_groups, costs.shape[:3])


def synchronize_handedness(
    async_graph: np.ndarray, *, reference_pair: tuple[int, int]
) -> np.ndarray:
    if (
        async_graph.ndim != 4
        or async_graph.shape[-2:] != (3, 3)
        or async_graph.shape[0] != async_graph.shape[1]
    ):
        raise ValueError(
            "async_graph must be an array of shape (n_images, n_images, 3, 3)"
        )

    n_images = async_graph.shape[0]
    if (
        not all(0 <= k < n_images for k in reference_pair)
        or reference_pair[0] == reference_pair[1]
    ):
        raise ValueError(
            f"Provided an invalid reference pair. Expected a pair of distinct "
            f"integers in [0, {n_images = }), got {reference_pair}"
        )

    graph = np.array(async_graph, dtype=np.float64, copy=True)
    J = HANDEDNESS_REFLECTION

    diagonal = np.arange(n_images)
    graph[diagonal, diagonal] = np.eye(3)

    if n_images == 2:
        reference_i, reference_j = reference_pair

        predicted = graph[reference_i, reference_j].T
        measured = graph[reference_j, reference_i]
        mirrored = J @ measured @ J

        direct_error = np.linalg.norm(predicted - measured, ord="fro")
        mirrored_error = np.linalg.norm(predicted - mirrored, ord="fro")

        if mirrored_error < direct_error:
            graph[reference_j, reference_i] = mirrored

        return graph

    matrix = JSyncAdjacencyMatrix.from_async_graph(async_graph=graph)

    eigenvector = matrix.leading_eigenvector()

    # k = 1 excludes diagonal self-loops from the comparison
    i, j = np.triu_indices(graph.shape[0], k=1)
    order = matrix.pair_idx(i, j)

    # Use the reference pair to fix one of the two global handedness solutions
    reference_pair_index = matrix.pair_idx(*reference_pair)
    negative = eigenvector < 0
    upper_reflection_mask = negative[order] != negative[reference_pair_index]

    # J-sync upper triangular entries
    if upper_reflection_mask.any():
        selected = graph[i[upper_reflection_mask], j[upper_reflection_mask]]
        graph[i[upper_reflection_mask], j[upper_reflection_mask]] = J @ selected @ J

    # Cache upper matrices to avoid re-slicing
    upper_matrices = graph[i, j]

    # Synchronize lower triangular entries (j, i) to match upper entries (i, j)
    lower_transposed = graph[j, i].swapaxes(-2, -1)

    # Transposing and mirroring commute, avoid calculating things twice
    mirrored_lower_transposed = J @ lower_transposed @ J

    direct_errors = np.linalg.norm(
        lower_transposed - upper_matrices, ord="fro", axis=(-2, -1)
    )
    mirrored_errors = np.linalg.norm(
        mirrored_lower_transposed - upper_matrices, ord="fro", axis=(-2, -1)
    )

    lower_reflection_mask = mirrored_errors < direct_errors

    if lower_reflection_mask.any():
        selected = graph[j[lower_reflection_mask], i[lower_reflection_mask]]
        graph[j[lower_reflection_mask], i[lower_reflection_mask]] = J @ selected @ J

    # If the reference pair was a lower-triangle pair, ensure its handedness is preserved
    # NOTE: relies on alignment between np.triu_indices and JSyncAdjacencyMatrix.pair_idx()
    if (
        reference_pair[0] > reference_pair[1]
        and lower_reflection_mask[reference_pair_index]
    ):
        graph = J @ graph @ J

    return graph
