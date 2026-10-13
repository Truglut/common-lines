from dataclasses import dataclass
import itertools

import numpy as np

from common_lines.alignment import (
    HANDEDNESS_REFLECTION,
    get_transformation_for_second_image,
)
from common_lines.centering import precenter_images, PrecenteringMethod
from common_lines.centering.preprocessing import (
    PreprocessingOptions,
    default_preprocessing_options,
    preprocess_images,
)
from common_lines.detection import detect_common_lines_all_pairs
from common_lines.procrustes import nearest_rotation
from common_lines.radon import batch_common_line_sinograms

DEFAULT_THETA = np.arange(0.0, 180.0, 0.5)


@dataclass(frozen=True, slots=True)
class OrientationEstimationResult:
    """
    Relative orientation graph plus estimation diagnostics.

    Attributes
    ----------
    graph: numpy.ndarray
        Estimated directed relative-orientation graph with shape
        ``(n_images, n_images, 3, 3)``.

        ``graph[i, j]`` transforms vector coordinates from the coordinate system
        of image ``i`` to that of image ``j``:

            v_j = graph[i, j] @ v_i

        The graph contains one of the two globally mirrored handedness solutions.
        The other is obtained with
        ``mirror_relative_orientations_graph(graph)``.
    used_fallback: numpy.ndarray
        Boolean array of shape ``(n_images, n_images)``, indicating image pairs
        where the estimated relative orientation was resolved using a fallback
        method after no valid third image was found for angular reconstitution.
    handedness_reference: tuple[int, int]
        Image pair used to fix one consistent handedness for the whole graph.
    """

    graph: np.ndarray
    used_fallback: np.ndarray
    handedness_reference: tuple[int, int]


def build_relative_orientations_graph(
    images: np.ndarray,
    *,
    n_third: int = 1,
    random_seed: int | None = 0,
    handedness_reference: tuple[int, int] | None = None,
    theta: np.ndarray | None = None,
    precenter_method: PrecenteringMethod | None = "center_of_mass",
    common_line_block_size: int = 4,
    preprocessing_options: PreprocessingOptions | None = None,
) -> np.ndarray:
    """
    Build a graph of relative 3D orientations between all image pairs.

    Parameters
    ----------
    images : np.ndarray
        Input images with shape ``(n_images, height, width)``.
    n_third : int
        Maximum number of valid third images used to estimate each ordered pair.
        Must lie in ``[1, n_images - 2]``.
    random_seed : int, optional
        Seed used to randomize the order in which third images are tested.
    handedness_reference : tuple[int, int], optional
        Ordered image pair used to fix one of the two global handedness
        solutions. Its first candidate is selected arbitrarily, and the
        remaining branches are chosen consistently with it.

        If ``None``, an edge estimated without fallback is selected
        automatically when possible. If every edge used the fallback, the
        particular reference is irrelevant and ``(0, 1)`` is used.
    theta : np.ndarray, optional
        Angular grid in degrees used to compute the sinograms. If None,
        ``DEFAULT_THETA`` is used.
    precenter_method : PrecenteringMethod, optional
        Method used to estimate particle translations before common-line
        detection. ``"center_of_mass"`` is used by default. Pass ``None`` to
        disable precentering.
    common_line_block_size : int, optional
        Number of images per correlation block in all-pairs common-line
        detection. Larger values may improve throughput but require more memory.
        Default is 4.
    preprocessing_options : PreprocessingOptions, optional
        Preprocessing used both to estimate precentering shifts and, after
        centering, before common-line detection. ``None`` selects the standard
        circular-mask/background-subtraction defaults. 
        Pass ``PreprocessingOptions()`` to disable preprocessing in both stages 
        (i.e., use the raw input images both to estimate the precentering shifts and
        their precentered version for common-line detection; this does not disable 
        precentering itself).

    Returns
    -------
    numpy.ndarray
        Directed relative-orientation graph with shape
        ``(n_images, n_images, 3, 3)``.

        ``graph[i, j]`` transforms vector coordinates from image ``i`` to image
        ``j``:

            v_j = graph[i, j] @ v_i

        Equivalently, if ``R_i`` and ``R_j`` transform coordinates from a common
        reference system to images ``i`` and ``j``, then ideally

            graph[i, j] = R_j @ R_i.T

    Notes
    -----
    The pairwise rotations are estimated independently. The graph is not forced
    to satisfy inverse or cycle consistency.

    The opposite, equally valid global solution is obtained by conjugating every
    graph block with ``HANDEDNESS_REFLECTION``. This can be done with the
    ``mirror_relative_orientations_graph`` function.
    """
    return build_relative_orientations_graph_with_diagnostics(
        images,
        n_third=n_third,
        random_seed=random_seed,
        handedness_reference=handedness_reference,
        theta=theta,
        precenter_method=precenter_method,
        common_line_block_size=common_line_block_size,
        preprocessing_options=preprocessing_options,
    ).graph


def build_relative_orientations_graph_with_diagnostics(
    images: np.ndarray,
    *,
    n_third: int = 1,
    random_seed: int | None = 0,
    handedness_reference: tuple[int, int] | None = None,
    theta: np.ndarray | None = None,
    precenter_method: PrecenteringMethod | None = "center_of_mass",
    common_line_block_size: int = 4,
    preprocessing_options: PreprocessingOptions | None = None,
) -> OrientationEstimationResult:
    """
    Estimate the directed relative-orientation graph and return diagnostics.

    This function uses the same inputs and estimation pipeline as
    ``build_relative_orientations_graph``, but additionally reports which
    edges required the pairwise fallback and which edge was used to fix the
    global handedness.

    The graph follows the same convention as
    ``build_relative_orientations_graph``:

        v_j = graph[i, j] @ v_i

    for coordinates expressed in the systems of images ``i`` and ``j``.

    Returns
    -------
    OrientationEstimationResult
        Orientation graph and associated estimation diagnostics.
    """
    images, theta_array = _validate_pipeline_inputs(
        images,
        n_third=n_third,
        handedness_reference=handedness_reference,
        theta=theta,
        common_line_block_size=common_line_block_size,
    )

    resolved_preprocessing = (
        default_preprocessing_options(images)
        if preprocessing_options is None
        else preprocessing_options
    )

    if precenter_method is not None:
        images = precenter_images(
            images,
            method=precenter_method,
            preprocessing_options=resolved_preprocessing,
        )

    images_for_detection = preprocess_images(images, options=resolved_preprocessing)
    sinograms = batch_common_line_sinograms(images_for_detection, theta=theta_array)
    del images, images_for_detection

    common_lines = detect_common_lines_all_pairs(
        sinograms, allow_reversed=True, block_size=common_line_block_size
    )
    angle_1, angle_2 = common_lines.oriented_angles(theta_array)
    del sinograms, common_lines

    candidates, used_fallback = _estimate_relative_orientation_candidates(
        angle_1, angle_2, n_third=n_third, random_seed=random_seed
    )
    del angle_1, angle_2

    reference_pair = _resolve_handedness_reference(
        used_fallback, requested=handedness_reference
    )
    graph = _select_consistent_handedness(candidates, reference_pair=reference_pair)

    return OrientationEstimationResult(
        graph=graph,
        used_fallback=used_fallback,
        handedness_reference=reference_pair,
    )


def mirror_relative_orientations_graph(graph: np.ndarray) -> np.ndarray:
    """
    Return the opposite-handed solution for a relative-orientation graph.

    Every relative rotation is conjugated with

        J = diag(1, 1, -1)

    so that

        mirrored[i, j] = J @ graph[i, j] @ J.

    The returned graph follows the same coordinate convention as the input:

        v_j = mirrored[i, j] @ v_i

    Parameters
    ----------
    graph
        Relative-orientation graph with shape
        ``(n_images, n_images, 3, 3)``.

    Returns
    -------
    numpy.ndarray
        Graph representing the globally mirrored handedness solution.
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
    handedness_reference: tuple[int, int] | None,
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

    if handedness_reference is not None:
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


def _resolve_handedness_reference(
    used_fallback: np.ndarray, *, requested: tuple[int, int] | None
) -> tuple[int, int]:
    """Use an informative non-fallback edge as the default handedness reference."""
    if requested is not None:
        return requested

    n_images = used_fallback.shape[0]
    for i, j in itertools.permutations(range(n_images), 2):
        if not used_fallback[i, j]:
            return i, j

    # If every edge used the in-plane fallback, the two handedness branches are
    # identical everywhere, so the particular reference pair is immaterial.
    return 0, 1


def _estimate_relative_orientation_candidates(
    angle_1: np.ndarray,
    angle_2: np.ndarray,
    *,
    n_third: int,
    random_seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate both handedness candidates for every ordered image pair.

    Parameters
    ----------
    angle_1, angle_2
        Oriented common-line angle matrices with shape
        ``(n_images, n_images)``. For pair ``(i, j)``,
        ``angle_1[i, j]`` is the common-line angle in image ``i`` and
        ``angle_2[i, j]`` is the corresponding oriented angle in image ``j``.
        Angles are expressed in degrees.
    n_third
        Maximum number of valid third images used for each ordered pair.
    random_seed
        Seed used to randomize the order of candidate third images.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_images, n_images, 2, 3, 3)``. For each ordered pair
        ``(i, j)``, ``candidates[i, j]`` contains the two handedness candidates
        for the coordinate transformation from image ``i`` to image ``j``.
    numpy.ndarray
        Boolean array with shape ``(n_images, n_images)`` indicating for each directed
        image pair, whether a fallback was used due to not finding a valid third
        image for the angular reconstitution algorithm.
    """
    n_images = angle_1.shape[0]
    image_indices = np.arange(n_images)
    rng = np.random.default_rng(random_seed)

    candidates = np.empty(
        (n_images, n_images, 2, 3, 3),
        dtype=np.float64,
    )
    candidates[image_indices, image_indices, :, :, :] = np.eye(3)
    used_fallback = np.zeros((n_images, n_images), dtype=bool)

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
                direct_r2, mirrored_r2 = get_transformation_for_second_image(
                    angle_12_in_1_deg=float(angle_1[i, j]),
                    angle_12_in_2_deg=float(angle_2[i, j]),
                    angle_13_in_1_deg=float(angle_1[i, k]),
                    angle_13_in_3_deg=float(angle_2[i, k]),
                    angle_23_in_2_deg=float(angle_1[j, k]),
                    angle_23_in_3_deg=float(angle_2[j, k]),
                )
            except ValueError:
                continue

            # Image i is the reference image for this triplet, so R2 transforms
            # coordinates from image i to image j.
            branch_pairs.append((direct_r2, mirrored_r2))
            if len(branch_pairs) == n_third:
                break

        if not branch_pairs:
            fallback_pair = _estimate_pairwise_fallback(angle_1[i, j], angle_2[i, j])
            branch_pairs.append(fallback_pair)
            used_fallback[i, j] = True

        direct_mean, mirrored_mean = _average_handedness_branches(branch_pairs)

        candidates[i, j, 0] = direct_mean
        candidates[i, j, 1] = mirrored_mean

    return candidates, used_fallback


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
                # With graph[a, b] mapping a -> b, the path
                # reference_i -> reference_j -> k is
                # R_{j->k} @ R_{i->j}.
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

        # R_{i->j} = R_{ref->j} @ R_{ref->i}^{-1}.
        # Rotation matrices are orthogonal, so the inverse is the transpose.
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


def _estimate_pairwise_fallback(
    theta_1_deg: float, theta_2_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a canonical relative orientation from one pairwise common line.

    A single common line does not determine the full 3D relative orientation.
    We therefore choose the minimum-angle rotation that maps the coordinates of 
    common-line direction in image 1 onto the corresponding coordinates in image 2.
    Since both directions lie in their image XY planes, this is a rotation about Z
    by ``theta_2_deg - theta_1_deg``.

    The mirrored handedness candidate is identical for this pure in-plane
    rotation, so the same matrix is returned for both branches.

    Parameters
    ----------
    theta_1_deg : float
        Oriented common-line angle in image 1, in degrees.
    theta_2_deg : float
        Corresponding oriented common-line angle in image 2, in degrees.

    Returns
    -------
    numpy.ndarray
        Canonical rotation mapping coordinates from image 1 to image 2.
    numpy.ndarray
        Mirrored handedness candidate, equal to the first matrix.
    """
    delta_rad = np.deg2rad(theta_2_deg - theta_1_deg)
    cos_delta = np.cos(delta_rad)
    sin_delta = np.sin(delta_rad)

    rotation = np.array(
        [
            [cos_delta, -sin_delta, 0.0],
            [sin_delta, cos_delta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rotation, rotation
