"""Estimate both handedness candidates per image pair using third-image witnesses."""

import itertools
from typing import Literal

import numpy as np

from common_lines.geometry.euler import EulerAngles, euler_zyz_to_matrix
from common_lines.geometry.procrustes import nearest_rotation
from common_lines.reconstitution.euler import (
    alignment_euler_angles,
    rot_psi_from_angle_pair,
    valid_euler_triplets,
    vote_tilt,
)
from common_lines.reconstitution.triplets import (
    HANDEDNESS_REFLECTION,
    CommonLineAngles,
    CommonLineTriplet,
)

ReconstitutionMethod = Literal["averaging", "voting"]


def _average_handedness_branches(
    all_direct: np.ndarray,
    all_mirrored: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Average triplet estimates after aligning their handedness branches.

    The first triplet fixes an arbitrary local hand. For each following triplet,
    the branch closest to the current mean is selected before updating the mean.
    """
    selected = [np.asarray(all_direct[0], dtype=np.float64)]
    mean_rotation = selected[0]

    for direct, mirrored in zip(all_direct[1:], all_mirrored[1:]):
        direct_distance = np.linalg.norm(direct - mean_rotation, ord="fro")
        mirrored_distance = np.linalg.norm(mirrored - mean_rotation, ord="fro")

        selected.append(direct if direct_distance <= mirrored_distance else mirrored)
        mean_rotation = nearest_rotation(np.mean(selected, axis=0))

    direct_mean = nearest_rotation(np.mean(selected, axis=0))
    J = HANDEDNESS_REFLECTION
    return direct_mean, J @ direct_mean @ J


def _estimate_pairwise_fallback(
    theta_1_rad: float, theta_2_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a canonical relative orientation from one pairwise common line.

    A single common line does not determine the full 3D relative orientation.
    We therefore choose the minimum-angle rotation that maps the coordinates of
    common-line direction in image 1 onto the corresponding coordinates in image 2.
    Since both directions lie in their image XY planes, this is a rotation about Z
    by ``theta_2_rad - theta_1_rad``.

    The mirrored handedness candidate is identical for this pure in-plane
    rotation, so the same matrix is returned for both branches.

    Parameters
    ----------
    theta_1_rad : float
        Oriented common-line angle in image 1, in radians.
    theta_2_rad : float
        Corresponding oriented common-line angle in image 2, in radians.

    Returns
    -------
    numpy.ndarray
        Canonical rotation mapping coordinates from image 1 to image 2.
    numpy.ndarray
        Mirrored handedness candidate, equal to the first matrix.
    """
    delta_rad = theta_2_rad - theta_1_rad
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


def estimate_relative_orientation_candidates(
    angle_1: np.ndarray,
    angle_2: np.ndarray,
    *,
    n_third: int,
    random_seed: int | None,
    method: ReconstitutionMethod,
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
        Angles are expressed in radians.
    n_third
        Maximum number of valid third images used for each ordered pair.
    random_seed
        Seed used to sample valid third images uniformly without replacement.
    method
        "voting" estimates a Gaussian tilt mode with a five-degree
        bandwidth and continuous refinement. "averaging" averages rotations
        after aligning their handedness branches.

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
    if method not in ("averaging", "voting"):
        raise ValueError("method must be 'averaging' or 'voting'.")
    if n_third < 1:
        raise ValueError("n_third must be at least 1.")

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

    def pair_at(i: int | np.ndarray, j: int | np.ndarray) -> CommonLineAngles:
        return CommonLineAngles(first_rad=angle_1[i, j], second_rad=angle_2[i, j])

    for i, j in itertools.permutations(range(n_images), 2):
        available_mask[:] = True
        available_mask[i] = False
        available_mask[j] = False

        third_indices = image_indices[available_mask]
        triplets = CommonLineTriplet(
            one_with_two=pair_at(i, j),
            one_with_three=pair_at(i, third_indices),
            two_with_three=pair_at(j, third_indices),
        )
        valid_indices = third_indices[valid_euler_triplets(triplets)]

        if valid_indices.size:
            selected = rng.choice(
                valid_indices, size=min(n_third, valid_indices.size), replace=False
            )
            selected_triplets = CommonLineTriplet(
                one_with_two=triplets.one_with_two,
                one_with_three=pair_at(i, selected),
                two_with_three=pair_at(j, selected),
            )

            if method == "averaging":
                # Filtering and solving share the same geometry checks. A solver
                # failure here is unexpected and must propagate to the caller.
                direct, mirrored = alignment_euler_angles(selected_triplets)
                direct_mean, mirrored_mean = _average_handedness_branches(
                    euler_zyz_to_matrix(direct), euler_zyz_to_matrix(mirrored)
                )
            elif method == "voting":
                consensus_tilt = vote_tilt(
                    selected_triplets, bins=181, sigma=np.deg2rad(5.0)
                )
                rot, psi = rot_psi_from_angle_pair(angle_1[i, j], angle_2[i, j])

                direct_euler = EulerAngles(rot, consensus_tilt, psi)
                mirrored_euler = EulerAngles(rot, -consensus_tilt, psi)

                direct_mean, mirrored_mean = (
                    euler_zyz_to_matrix(direct_euler),
                    euler_zyz_to_matrix(mirrored_euler),
                )
        else:
            direct_mean, mirrored_mean = _estimate_pairwise_fallback(
                angle_1[i, j], angle_2[i, j]
            )
            used_fallback[i, j] = True

        candidates[i, j, 0] = direct_mean
        candidates[i, j, 1] = mirrored_mean

    return candidates, used_fallback
