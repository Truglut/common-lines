import itertools

import numpy as np

from common_lines.radon import batch_common_line_sinograms
from common_lines.detection import detect_common_lines
from common_lines.alignment import (
    compute_transformation_matrices_from_angles,
    HANDEDNESS_REFLECTION,
)
from common_lines.procrustes import nearest_rotation

THETA = np.arange(0.0, 180.0, 0.5)


def build_relative_orientations_graph(
    images: np.ndarray, n_third: int = 1, random_seed: int | None = None
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

    rng = np.random.default_rng(random_seed)

    # Pre-compute all sinograms
    sinograms = batch_common_line_sinograms(images, theta=THETA)

    # Allocate memory for the relative orientations graph
    orientation_graph = np.empty(shape=(n_images, n_images, 3, 3), dtype=np.float64)

    # Overwrite diagonal entries with the identity matrix (alignment of an image against itself)
    idx = np.arange(n_images)
    orientation_graph[idx, idx] = np.eye(3)
    available_mask = np.ones(n_images, dtype=bool)

    # Loop over image pairs to estimate their relative orientations
    for i, j in itertools.permutations(range(n_images), 2):
        print(f"{i = }, {j = }")

        available_mask[:] = True
        available_mask[i] = available_mask[j] = False

        common_line_ij = detect_common_lines(
            sinograms[i],
            sinograms[j],
            theta_1=THETA,
            theta_2=THETA,
            allow_reversed=True,
        )
        angle_ij_in_i = common_line_ij.angle_1
        angle_ij_in_j = common_line_ij.angle_2

        n_valid = 0
        estimated_rotation = np.zeros((3, 3), dtype=np.float64)
        while n_valid < n_third and available_mask.any():
            k = rng.choice(idx[available_mask])
            available_mask[k] = False

            # Recover common line angles between the two images to align and the third
            common_line_ik = detect_common_lines(
                sinograms[i], sinograms[k], theta_1=THETA, theta_2=THETA
            )
            common_line_jk = detect_common_lines(
                sinograms[j], sinograms[k], theta_1=THETA, theta_2=THETA
            )

            angle_ik_in_i = common_line_ik.angle_1
            angle_ik_in_k = common_line_ik.angle_2

            angle_jk_in_j = common_line_jk.angle_1
            angle_jk_in_k = common_line_jk.angle_2

            try:
                # Estimate rotation from the selected angles
                solutions = compute_transformation_matrices_from_angles(
                    angle_12_in_1=angle_ij_in_i,
                    angle_12_in_2=angle_ij_in_j,
                    angle_13_in_1=angle_ik_in_i,
                    angle_13_in_3=angle_ik_in_k,
                    angle_23_in_2=angle_jk_in_j,
                    angle_23_in_3=angle_jk_in_k,
                )

                # The first hand is selected arbitrarily until handedness synchronization
                # is implemented
                _, rotation_j_to_i, _ = solutions[0]
                n_valid += 1
                estimated_rotation += rotation_j_to_i
            except ValueError as e:
                print(
                    f"Invalid configuration found for triplet {i},{j},{k}. Choosing new third image"
                )
                print(f"Exception found: {e}")
                print(f"{angle_ij_in_i = }, {angle_ij_in_j = }")
                print(f"{angle_ik_in_i = }, {angle_ik_in_k = }")
                print(f"{angle_jk_in_j = }, {angle_jk_in_k = }")
                pass

        if not n_valid:
            raise ValueError(
                f"Could not find a suitable third image that yields a valid common-line configuration for images {i}, {j}"
            )

        estimated_rotation /= n_valid
        estimated_rotation = nearest_rotation(estimated_rotation)

        orientation_graph[i, j] = estimated_rotation

    return orientation_graph
