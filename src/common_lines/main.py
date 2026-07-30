import itertools

import numpy as np

from common_lines.radon import batch_common_line_sinograms
from common_lines.detection import detect_common_lines_all_pairs
from common_lines.alignment import compute_transformation_matrices_from_angles
from common_lines.procrustes import nearest_rotation

THETA = np.arange(0.0, 360.0, 0.5)


def build_relative_orientations_graph(
    images: np.ndarray, n_third: int = 1, random_seed: int | None = None
) -> np.ndarray:
    """
    Builds a graph of relative 3D orientations between all pairs of images.

    This function estimates the relative rotation matrix between every pair of 
    images using a common-lines approach. Since a unique relative orientation 
    requires a triplet of images to be resolved, the algorithm anchors each pair 
    $(i, j)$ with a user-defined number of 'third images' ($k$). 

    For each triplet $(i, j, k)$, a candidate rotation matrix is computed. The 
    final relative orientation for the pair is estimated by averaging these 
    matrices and projecting the result back onto the special orthogonal group 
    $SO(3)$ (finding the closest valid rotation matrix in Frobenius norm via SVD).

    Parameters
    ----------
    images : np.ndarray
        Array of input images with shape `(n_images, height, width)`.
    n_third : int, optional
        Number of random 'third images' to sample as anchors for each pair. 
        Must be between 1 and `n_images - 2`. By default 1.
    random_seed : int or None, optional
        Seed for the random number generator to ensure reproducible triplet 
        selection. By default None.

    Returns
    -------
    np.ndarray
        The relative orientations graph as an array `A` of shape 
        `(n_images, n_images, 3, 3)`. The element `A[i, j]` is a $3 \times 3$ 
        rotation matrix that aligns image `j` to the coordinate system of image `i`.
        The diagonal elements `A[i, i]` are initialized as identity matrices.

    Raises
    ------
    ValueError
        If `n_third` is not in the valid range $[1, n_{images} - 2]$.
    """
    n_images = images.shape[0]

    # Validate input: number of third images must be compatible with dataset
    if not (1 <= n_third <= n_images - 2):
        raise ValueError(
            f"n_third must be in [1, n_images - 2] = [1, {n_images - 2}], got {n_third}."
        )

    rng = np.random.default_rng(random_seed)

    # Compute sinograms and detect common lines from them
    sinograms = batch_common_line_sinograms(images, theta=THETA)
    common_lines_all_pairs = detect_common_lines_all_pairs(sinograms, theta=THETA)
    del sinograms

    # Allocate memory for the relative orientations graph
    orientation_graph = np.empty(shape=(n_images, n_images, 3, 3), dtype=np.float64)

    # Overwrite diagonal entries with the identity matrix (alignment of an image against itself)
    idx = np.arange(n_images)
    orientation_graph[idx, idx] = np.eye(3)

    # Loop over image pairs to estimate their relative orientations
    for i, j in itertools.permutations(range(n_images), 2):
        # Select random third images to compute the relative orientation of j against i
        available_indices = idx[(idx != i) & (idx != j)]
        third_images = rng.choice(available_indices, size=n_third, replace=False)

        # Recover common line angles between the two images to align
        angle_ij_in_i = common_lines_all_pairs.angle_1[i, j]
        angle_ij_in_j = common_lines_all_pairs.angle_2[i, j]

        estimated_rotation = np.zeros((3, 3), dtype=np.float64)
        for k in third_images:
            # Recover common line angles between the two images to align and the third
            angle_ik_in_i = common_lines_all_pairs.angle_1[i, k]
            angle_ik_in_k = common_lines_all_pairs.angle_2[i, k]

            angle_jk_in_j = common_lines_all_pairs.angle_1[j, k]
            angle_jk_in_k = common_lines_all_pairs.angle_2[j, k]

            # Estimate rotation from the selected angles
            _, rotation_j_to_i, _ = compute_transformation_matrices_from_angles(
                angle_12_in_1=angle_ij_in_i,
                angle_12_in_2=angle_ij_in_j,
                angle_13_in_1=angle_ik_in_i,
                angle_13_in_3=angle_ik_in_k,
                angle_23_in_2=angle_jk_in_j,
                angle_23_in_3=angle_jk_in_k,
            )
            estimated_rotation += rotation_j_to_i

        # Average all the estimated rotations and project back to a valid rotations
        estimated_rotation /= n_third
        estimated_rotation = nearest_rotation(estimated_rotation)

        # Save the results
        orientation_graph[i, j] = estimated_rotation

    return orientation_graph
