import itertools

import numpy as np

from common_lines.radon import batch_common_line_sinograms
from common_lines.detection import detect_common_lines, CommonLineResult
from common_lines.alignment import (
    compute_transformation_matrices,
    HANDEDNESS_REFLECTION,
)
from common_lines.procrustes import nearest_rotation

THETA = np.arange(0.0, 180.0, 0.5)


def build_relative_orientations_graph(
    images: np.ndarray,
    n_third: int = 1,
    random_seed: int | None = None,
    handedness_reference: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """
    Builds a graph of relative 3D orientations between all pairs of images.

    This function estimates the relative rotation matrix between every pair of
    images using a common-lines approach. Since a unique relative orientation
    requires a triplet of images to be resolved, the algorithm anchors each pair
    $(i, j)$ with a user-defined number of 'third images' ($k$).

    When several third images are used, the two handedness branches returned for
    each triplet are aligned before averaging. A final, simple handedness-selection
    step chooses one branch per ordered pair using triangle consistency with a fixed
    pair of reference images.

    The selected matrices are still the independently estimated pairwise
    rotations. They are not replaced by products of other graph edges, and the
    graph is not forced to satisfy inverse or cycle consistency.

    Parameters
    ----------
    images
        Input images with shape ``(n_images, height, width)``.
    n_third
        Maximum number of valid third images used to estimate each ordered pair.
        Must lie in ``[1, n_images - 2]``.
    random_seed
        Seed used to randomize the order in which third images are tested.
    handedness_reference
        Ordered pair used to choose one of the two global handedness solutions.
        Its first candidate is selected arbitrarily, and the remaining branches
        are chosen relative to it.

    Returns
    -------
    numpy.ndarray
        Directed relative-orientation graph with shape
        ``(n_images, n_images, 3, 3)``. The block ``A[i, j]`` aligns image
        ``j`` with the coordinate system of image ``i``.

    Notes
    -----
    The opposite, equally valid global solution is obtained by conjugating every
    graph block with ``HANDEDNESS_REFLECTION``.
    """
    # Input validation: images shape and number of third images
    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError(
            "images must have shape (n_images, height, width). "
            f"Got array with shape {images.shape}."
        )
    n_images = images.shape[0]
    if n_images < 3:
        raise ValueError("At least three images are required.")

    if not (1 <= n_third <= n_images - 2):
        raise ValueError(
            f"n_third must be in [1, n_images - 2] = [1, {n_images - 2}], got {n_third}."
        )

    # Select the given handedness reference and validate it
    reference_i, reference_j = handedness_reference
    if not (
        0 <= reference_i < n_images
        and 0 <= reference_j < n_images
        and reference_i != reference_j
    ):
        raise ValueError(
            "handedness_reference must contain two different valid image indices."
        )

    candidates = _estimate_relative_orientation_candidates(
        images, n_third=n_third, random_seed=random_seed
    )

    return _select_consistent_handedness(
        candidates, reference_pair=handedness_reference
    )


def mirror_relative_orientations_graph(graph: np.ndarray) -> np.ndarray:
    """
    Return the opposite-handed version of a relative-orientation graph.

    Parameters
    ----------
    graph
        Array with shape ``(n_images, n_images, 3, 3)``.

    Returns
    -------
    numpy.ndarray
        Graph obtained by conjugating every block with
        ``HANDEDNESS_REFLECTION``.
    """
    graph = np.asarray(graph, dtype=np.float64)
    if graph.ndim != 4 or graph.shape[2:] != (3, 3):
        raise ValueError(
            "graph must have shape (n_images, n_images, 3, 3). "
            f"Got array with shape {graph.shape}."
        )
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("The first two graph dimensions must be equal.")

    J = HANDEDNESS_REFLECTION
    return J @ graph @ J


def _estimate_relative_orientation_candidates(
    images: np.ndarray, *, n_third: int, random_seed: int
) -> np.ndarray:
    n_images = images.shape[0]
    rng = np.random.default_rng(random_seed)

    # Pre-compute all sinograms
    sinograms = batch_common_line_sinograms(images, theta=THETA)
    common_line_cache: dict[tuple[int, int], CommonLineResult] = {}

    def get_common_line(i: int, j: int) -> CommonLineResult:
        key = (i, j)
        if key not in common_line_cache:
            common_line_cache[key] = detect_common_lines(
                sinograms[i],
                sinograms[j],
                theta_1=THETA,
                theta_2=THETA,
                allow_reversed=True,
            )
        return common_line_cache[key]

    # Allocate memory for the relative orientations graph
    candidates = np.empty(shape=(n_images, n_images, 2, 3, 3), dtype=np.float64)

    # Overwrite diagonal entries with the identity matrix (alignment of an image against itself)
    image_indices = np.arange(n_images)
    candidates[image_indices, image_indices, 0] = np.eye(3)
    candidates[image_indices, image_indices, 1] = np.eye(3)

    available_mask = np.ones(n_images, dtype=bool)
    # Loop over image pairs to estimate their relative orientations
    for i, j in itertools.permutations(range(n_images), 2):
        # Select available indices for third images and randomly shuffle them
        available_mask[:] = True
        available_mask[i] = available_mask[j] = False
        third_indices = image_indices[available_mask].copy()
        rng.shuffle(third_indices)

        common_line_ij = get_common_line(i, j)
        branch_pairs: list[tuple[np.ndarray, np.ndarray]] = []

        # Iterate over possible third images to estimate orientations
        for k in third_indices:
            common_line_ik = get_common_line(i, int(k))
            common_line_jk = get_common_line(j, int(k))

            # Try to estimate orientation from the selected (i, j, k) triplet
            try:
                direct, mirrored = compute_transformation_matrices(
                    common_line_ij,
                    common_line_ik,
                    common_line_jk,
                )
            except ValueError:
                continue

            # If the triplet was valid, append the result to the list
            branch_pairs.append((direct[1], mirrored[1]))
            if len(branch_pairs) == n_third:
                break

        if not branch_pairs:
            raise ValueError(
                "Could not find a valid third image for the ordered pair "
                f"({i}, {j})."
            )

        direct, mirrored = _average_handedness_branches(branch_pairs)
        candidates[i, j, 0] = direct
        candidates[i, j, 1] = mirrored

    return candidates


def _average_handedness_branches(
    branch_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Average triplet estimates after aligning their handedness branches.

    The first triplet fixes an arbitrary local hand. For each following triplet,
    the branch closest to the current mean is selected before updating the mean.
    """
    selected = [np.asarray(branch_pairs[0][0], dtype=np.float64)]
    mean_rotation = selected[0]

    for direct, mirrored in branch_pairs[1:]:
        direct_distance = np.linalg.norm(direct - mean_rotation, ord="fro")
        mirrored_distance = np.linalg.norm(mirrored - mean_rotation, ord="fro")

        selected.append(direct if direct_distance <= mirrored_distance else mirrored)
        mean_rotation = nearest_rotation(np.mean(selected, axis=0))

    direct_mean = nearest_rotation(np.mean(selected, axis=0))
    J = HANDEDNESS_REFLECTION
    return direct_mean, J @ direct_mean @ J


def _select_consistent_handedness(
    candidates: np.ndarray, *, reference_pair: tuple[int, int]
) -> np.ndarray:
    """
    Select one globally consistent handedness branch for each graph edge.

    The first branch of ``reference_pair`` fixes the arbitrary global hand. A
    reference star is then selected using triangles that contain both reference
    images. Every remaining directed edge keeps its independent estimate and
    only chooses the branch closest to the rotation predicted by that star.
    """
    # Prepare and validate input
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 5 or candidates.shape[2:] != (2, 3, 3):
        raise ValueError(
            "candidates must have shape (n_images, n_images, 2, 3, 3). "
            f"Got array with shape {candidates.shape}."
        )
    if candidates.shape[0] != candidates.shape[1]:
        raise ValueError("The first two candidate dimensions must be equal.")

    n_images = candidates.shape[0]
    reference_i, reference_j = reference_pair

    graph = np.empty((n_images, n_images, 3, 3), dtype=np.float64)
    diagonal = np.arange(n_images)
    graph[diagonal, diagonal] = np.eye(3)

    # The first candidate of the reference edge fixes one of the two global hands
    graph[reference_i, reference_j] = candidates[reference_i, reference_j, 0]

    # Select a handedness-consistent star around the first reference image
    for k in range(n_images):
        if k == reference_i or k == reference_j:
            continue

        best_error = np.inf
        best_reference_branch = 0

        for branch_ik in range(2):
            for branch_jk in range(2):
                predicted = (
                    candidates[reference_j, k, branch_jk]
                    @ graph[reference_i, reference_j]
                )
                error = np.linalg.norm(
                    predicted - candidates[reference_i, k, branch_ik], ord="fro"
                )

                if error < best_error:
                    best_error = error
                    best_reference_branch = branch_ik

        graph[reference_i, k] = candidates[reference_i, k, best_reference_branch]

    # Choose every other directed edge using the selected reference star. The
    # stored edge remains its independently estimated candidate.
    for i, j in itertools.permutations(range(n_images), 2):
        if i == reference_i:
            continue

        predicted = graph[reference_i, j] @ graph[reference_i, i].T

        direct_distance = np.linalg.norm(candidates[i, j, 0] - predicted, ord="fro")
        mirrored_distance = np.linalg.norm(candidates[i, j, 1] - predicted, ord="fro")

        branch = 0 if direct_distance <= mirrored_distance else 1
        graph[i, j] = candidates[i, j, branch]

    return graph
