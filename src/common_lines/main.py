import itertools

import numpy as np

from common_lines.alignment import HANDEDNESS_REFLECTION, get_alignment_for_second_image
from common_lines.centering import precenter_images, PrecenteringMethod
from common_lines.centering.preprocessing import PreprocessingOptions
from common_lines.detection import detect_common_lines_all_pairs
from common_lines.procrustes import nearest_rotation
from common_lines.radon import batch_common_line_sinograms

DEFAULT_THETA = np.arange(0.0, 180.0, 0.5)


def build_relative_orientations_graph(
    images: np.ndarray,
    *,
    n_third: int = 1,
    random_seed: int | None = None,
    handedness_reference: tuple[int, int] = (0, 1),
    theta: np.ndarray | None = None,
    precenter_method: PrecenteringMethod | None = None,
    common_line_block_size: int = 4,
    preprocessing_options: PreprocessingOptions | None = None,
) -> np.ndarray:
    """
    Build a graph of relative 3D orientations between all image pairs.

    Sinograms and pairwise common lines are computed before the triplet loop.
    Common-line detection is performed in image blocks, avoiding both repeated
    pairwise normalization and the full correlation tensor with shape
    ``(n_images, n_images, n_angles, n_angles)``.

    A unique relative orientation requires a triplet of images. Each ordered pair
    ``(i, j)`` is therefore estimated using up to ``n_third`` valid third images.
    When several estimates are available, their handedness branches are aligned
    before averaging. A final handedness-selection step chooses one branch per
    ordered pair using triangle consistency with a fixed reference pair.

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
    theta
        Angular grid in degrees used to compute the sinograms. If None,
        ``DEFAULT_THETA`` is used.
    precenter
        If True, approximately center all particles before computing their
        sinograms. Horizontal and vertical corrections are estimated in batch
        from correlations with reflected versions of the original images.
    common_line_block_size
        Number of images per correlation block in all-pairs common-line
        detection. Larger values may improve throughput but require more memory.
    return_timings
        If True, also return a ``PipelineTimings`` instance.
    preprocessing_options : PreprocessingOptions, optional
        Parameters for ``preprocess_images``. See ``PreprocessingOptions``
        for more information.

    Returns
    -------
    numpy.ndarray or tuple
        Directed relative-orientation graph with shape
        ``(n_images, n_images, 3, 3)``. The block ``graph[i, j]`` aligns image
        ``j`` with the coordinate system of image ``i``. If ``return_timings``
        is True, the graph and timing information are returned as a tuple.

    Notes
    -----
    The pairwise rotations are estimated independently. The graph is not forced
    to satisfy inverse or cycle consistency.

    The opposite, equally valid global solution is obtained by conjugating every
    graph block with ``HANDEDNESS_REFLECTION``. This can be done with the
    ``mirror_relative_orientations_graph`` function.
    """
    images, theta_array = _validate_pipeline_inputs(
        images,
        n_third=n_third,
        handedness_reference=handedness_reference,
        theta=theta,
        common_line_block_size=common_line_block_size,
    )

    # Pre-center the full image batch before computing the sinograms
    if precenter_method is not None:
        images = precenter_images(
            images, method=precenter_method, preprocessing_options=preprocessing_options
        )

    sinograms = batch_common_line_sinograms(images, theta=theta_array)
    # Images are no longer needed once we have the sinograms
    del images

    # Detect all pairwise common lines
    common_lines = detect_common_lines_all_pairs(
        sinograms, allow_reversed=True, block_size=common_line_block_size
    )
    angle_1, angle_2 = common_lines.oriented_angles(theta_array)

    # Sinograms and common line indices are no longer needed once we have the angles
    del sinograms, common_lines

    # Estimate the two handedness candidates for every directed edge
    candidates = _estimate_relative_orientation_candidates(
        angle_1, angle_2, n_third=n_third, random_seed=random_seed
    )
    del angle_1, angle_2

    graph = _select_consistent_handedness(
        candidates, reference_pair=handedness_reference
    )

    return graph


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
        Graph obtained by conjugating every block with ``HANDEDNESS_REFLECTION``.
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


def _validate_pipeline_inputs(
    images: np.ndarray,
    *,
    n_third: int,
    handedness_reference: tuple[int, int],
    theta: np.ndarray | None,
    common_line_block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the public pipeline inputs and prepare the angular grid."""
    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError(
            "images must have shape (n_images, height, width). "
            f"Got array with shape {images.shape}."
        )

    if images.shape[-2] != images.shape[-1]:
        raise ValueError(
            f"Only square images are currently supported; got image shape {images.shape}"
        )

    n_images = images.shape[0]
    if n_images < 3:
        raise ValueError("At least three images are required.")

    if not (1 <= n_third <= n_images - 2):
        raise ValueError(
            f"n_third must be in [1, n_images - 2] = [1, {n_images - 2}], "
            f"got {n_third}."
        )

    reference_i, reference_j = handedness_reference
    if not (
        0 <= reference_i < n_images
        and 0 <= reference_j < n_images
        and reference_i != reference_j
    ):
        raise ValueError(
            "handedness_reference must contain two different valid image indices."
        )

    if common_line_block_size < 1:
        raise ValueError(
            "common_line_block_size must be at least 1, "
            f"got {common_line_block_size}."
        )

    theta_array = np.asarray(
        DEFAULT_THETA if theta is None else theta,
        dtype=np.float64,
    )
    if theta_array.ndim != 1 or theta_array.size == 0:
        raise ValueError("theta must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(theta_array)):
        raise ValueError("theta must contain only finite values.")

    return images, theta_array


def _estimate_relative_orientation_candidates(
    angle_1: np.ndarray,
    angle_2: np.ndarray,
    *,
    n_third: int,
    random_seed: int | None,
) -> np.ndarray:
    """
    Estimate both handedness candidates for every ordered image pair.

    Parameters
    ----------
    angle_1, angle_2
        Oriented common-line angle matrices with shape
        ``(n_images, n_images)``. For pair ``(i, j)``,
        ``angle_1[i, j]`` is the common-line angle in image ``i`` and
        ``angle_2[i, j]`` is the corresponding oriented angle in image ``j``.
    n_third
        Maximum number of valid third images used for each ordered pair.
    random_seed
        Seed used to randomize the order of candidate third images.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_images, n_images, 2, 3, 3)`` containing
        the two handedness candidates for every directed image pair.
    """
    n_images = angle_1.shape[0]
    image_indices = np.arange(n_images)
    rng = np.random.default_rng(random_seed)

    candidates = np.empty(
        (n_images, n_images, 2, 3, 3),
        dtype=np.float64,
    )
    candidates[image_indices, image_indices, :, :, :] = np.eye(3)

    available_mask = np.ones(n_images, dtype=bool)

    for i, j in itertools.permutations(range(n_images), 2):
        available_mask[:] = True
        available_mask[i] = False
        available_mask[j] = False

        third_indices = image_indices[available_mask].copy()
        rng.shuffle(third_indices)

        branch_pairs: list[tuple[np.ndarray, np.ndarray]] = []

        for k_value in third_indices:
            k = int(k_value)

            try:
                direct_r2, mirrored_r2 = get_alignment_for_second_image(
                    angle_12_in_1_deg=float(angle_1[i, j]),
                    angle_12_in_2_deg=float(angle_2[i, j]),
                    angle_13_in_1_deg=float(angle_1[i, k]),
                    angle_13_in_3_deg=float(angle_2[i, k]),
                    angle_23_in_2_deg=float(angle_1[j, k]),
                    angle_23_in_3_deg=float(angle_2[j, k]),
                )
            except ValueError:
                continue

            # The second matrix aligns image j with the frame of image i.
            branch_pairs.append((direct_r2, mirrored_r2))
            if len(branch_pairs) == n_third:
                break

        if not branch_pairs:
            raise ValueError(
                "Could not find a valid third image for the ordered pair "
                f"({i}, {j})."
            )

        direct_mean, mirrored_mean = _average_handedness_branches(branch_pairs)

        candidates[i, j, 0] = direct_mean
        candidates[i, j, 1] = mirrored_mean

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

    # The first candidate of the reference edge fixes one of the two global hands.
    graph[reference_i, reference_j] = candidates[reference_i, reference_j, 0]

    # Select a handedness-consistent star around the first reference image.
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
                    predicted - candidates[reference_i, k, branch_ik],
                    ord="fro",
                )

                if error < best_error:
                    best_error = error
                    best_reference_branch = branch_ik

        graph[reference_i, k] = candidates[
            reference_i,
            k,
            best_reference_branch,
        ]

    # Choose every other directed edge using the selected reference star.
    for i, j in itertools.permutations(range(n_images), 2):
        if i == reference_i:
            continue

        predicted = graph[reference_i, j] @ graph[reference_i, i].T

        direct_distance = np.linalg.norm(
            candidates[i, j, 0] - predicted,
            ord="fro",
        )
        mirrored_distance = np.linalg.norm(
            candidates[i, j, 1] - predicted,
            ord="fro",
        )

        branch = 0 if direct_distance <= mirrored_distance else 1
        graph[i, j] = candidates[i, j, branch]

    return graph
