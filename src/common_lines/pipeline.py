import itertools
from dataclasses import dataclass

import numpy as np

from common_lines.centering import PrecenteringMethod, estimate_shifts, shift_images
from common_lines.detection import detect_common_lines_all_pairs
from common_lines.graph_estimation import (
    ReconstitutionMethod,
    estimate_relative_orientation_candidates,
    select_consistent_handedness,
)
from common_lines.preprocessing import PreprocessingOptions, preprocess_images
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
    precenter_method: PrecenteringMethod | None = None,
    preprocessing_options: PreprocessingOptions | None = None,
    common_line_block_size: int = 4,
    reconstitution_method: ReconstitutionMethod = "voting",
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
        detection. Options are "center_of_mass" and "reflection".
        ``None`` disables precentering. Default is ``None``.
    preprocessing_options : PreprocessingOptions, optional
        Preprocessing used both to estimate precentering shifts and, after
        centering, before common-line detection. ``None`` disables preprocessing in
        both stages (i.e., use the raw input images both to estimate the precentering
        shifts and their precentered version for common-line detection; this does not
        disable precentering itself).
        Default is ``None``.
        Pass ``PreprocessingOptions.default_from_images(images)`` for a sensible
        preprocessing.
    common_line_block_size : int, optional
        Number of images per correlation block in all-pairs common-line
        detection. Larger values may improve throughput but require more memory.
        Default is 4.

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
        reconstitution_method=reconstitution_method,
    ).graph


def build_relative_orientations_graph_with_diagnostics(
    images: np.ndarray,
    *,
    n_third: int = 1,
    random_seed: int | None = 0,
    handedness_reference: tuple[int, int] | None = None,
    theta: np.ndarray | None = None,
    precenter_method: PrecenteringMethod | None = None,
    preprocessing_options: PreprocessingOptions | None = None,
    common_line_block_size: int = 4,
    reconstitution_method: ReconstitutionMethod = "voting",
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

    if precenter_method is not None:
        images_for_centering = preprocess_images(images, options=preprocessing_options)
        shifts = estimate_shifts(images_for_centering, method=precenter_method)
        del images_for_centering

        images = shift_images(images, shifts)

    images_for_detection = preprocess_images(images, options=preprocessing_options)
    sinograms = batch_common_line_sinograms(images_for_detection, theta=theta_array)
    del images, images_for_detection

    common_lines = detect_common_lines_all_pairs(
        sinograms, allow_reversed=True, block_size=common_line_block_size
    )
    # Input angles are in degrees, transforming the grid to radians avoids having to
    # do it for every individual angle
    theta_array_rad = np.deg2rad(theta_array)
    angle_1, angle_2 = common_lines.oriented_angles(theta_array_rad, radians=True)
    del sinograms, common_lines

    candidates, used_fallback = estimate_relative_orientation_candidates(
        angle_1,
        angle_2,
        n_third=n_third,
        random_seed=random_seed,
        method=reconstitution_method,
    )
    del angle_1, angle_2

    reference_pair = _resolve_handedness_reference(
        used_fallback, requested=handedness_reference
    )
    graph = select_consistent_handedness(candidates, reference_pair=reference_pair)

    return OrientationEstimationResult(
        graph=graph,
        used_fallback=used_fallback,
        handedness_reference=reference_pair,
    )


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
            f"common_line_block_size must be at least 1, got {common_line_block_size}."
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
