"""Compute common line angles from a known angular assignment. Useful for validating common line detection."""

# This code is heavily inspired by Oier Lauzirika's sinovar; in particular
# https://github.com/oierlauzi/sinovar/blob/main/sinovar/geometry/common_lines.py,
# viewed on 20th October 2026

import numpy as np


def _pick_orthogonal_vector(v: np.ndarray) -> np.ndarray:
    """Choose a nonzero perpendicular vector for each nonzero ``(..., 3)`` input."""
    x = v[..., 0]
    y = v[..., 1]
    z = v[..., 2]

    zero = np.zeros_like(x)
    return np.where(
        (np.abs(x) > np.abs(z))[..., None],
        np.stack((-y, x, zero), axis=-1),
        np.stack((zero, -z, y), axis=-1),
    )


def find_common_lines(directions0: np.ndarray, directions1: np.ndarray) -> np.ndarray:
    """
    Return unit common-line directions for pairs of projection-plane normals.

    Parameters
    ----------
    directions0, directions1 : np.ndarray
        Nonzero real vectors with shape ``(..., 3)``, expressed in the same
        reference coordinate system. Leading dimensions must broadcast.

    Returns
    -------
    numpy.ndarray
        Unit vectors with shape ``broadcast_shape + (3,)``. The orientation
        follows ``cross(directions0, directions1)``. For parallel or antiparallel
        normals, the planes share no unique line; choose an arbitrary direction
        perpendicular to the first normal instead.
    """
    cross = np.cross(directions0, directions1)
    cross = np.where(
        np.isclose(np.sum(np.square(cross), axis=-1), 0.0)[..., None],
        _pick_orthogonal_vector(directions0),
        cross,
    )

    cross /= np.linalg.norm(cross, axis=-1, keepdims=True)

    return cross


def compute_intrinsic_angles(
    matrices: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Express reference-space directions as oriented angles in image planes.

    ``matrices`` has shape ``(..., 3, 3)`` and maps reference coordinates to
    image coordinates; ``direction`` has shape ``(..., 3)`` in that reference
    system. Leading dimensions broadcast. Return ``atan2(y, x)`` in radians,
    in [-pi, pi], with the broadcast leading shape. For directions outside an
    image plane, the angle describes their projection onto that plane.
    """
    vx = matrices[..., 0, :]
    vy = matrices[..., 1, :]

    px = np.vecdot(direction, vx, axis=-1)  # type: ignore
    py = np.vecdot(direction, vy, axis=-1)  # type: ignore

    return np.atan2(py, px)


def compute_intrinsic_common_line_angles(
    matrices0: np.ndarray,
    matrices1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute oriented common-line angles from known image frames.

    Parameters
    ----------
    matrices0, matrices1 : np.ndarray
        Rotation matrices with shape ``(..., 3, 3)`` mapping coordinates from
        the same reference system into the two image frames. Their third rows
        are the projection-plane normals. Leading dimensions must broadcast.

    Returns
    -------
    angle_0, angle_1 : numpy.ndarray
        Angles of the same oriented common line in the first and second image,
        in radians in [-pi, pi], with the broadcast leading shape. Image-plane
        coordinates are ``(x, y) = (column, row)``. For coincident planes, return
        angles for the arbitrary shared direction chosen by ``find_common_lines``.

    Notes
    -----
    This is a forward model for validation, not a detector operating on images.
    For all pairs in a frame stack ``frames``, pass ``frames[:, None]`` and
    ``frames[None, :]``. Coincident-plane angles, including self-pairs, do not
    identify a unique common line.
    """
    directions0 = matrices0[..., 2, :]
    directions1 = matrices1[..., 2, :]
    common_line_direction = find_common_lines(directions0, directions1)
    return (
        compute_intrinsic_angles(matrices0, common_line_direction),
        compute_intrinsic_angles(matrices1, common_line_direction),
    )
