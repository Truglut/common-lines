"""Compute common line angles from a known angular assignment. Useful for validating common line detection."""

# This code is heavily inspired by Oier Lauzirika's sinovar; in particular
# https://github.com/oierlauzi/sinovar/blob/main/sinovar/geometry/common_lines.py,
# viewed on 20th October 2026

import numpy as np


def _pick_orthogonal_vector(v: np.ndarray) -> np.ndarray:
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
    vx = matrices[..., 0, :]
    vy = matrices[..., 1, :]

    px = np.vecdot(direction, vx, axis=-1)  # type: ignore
    py = np.vecdot(direction, vy, axis=-1)  # type: ignore

    return np.atan2(py, px)


def compute_intrinsic_common_line_angles(
    matrices0: np.ndarray,
    matrices1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    directions0 = matrices0[..., 2, :]
    directions1 = matrices1[..., 2, :]
    common_line_direction = find_common_lines(directions0, directions1)
    return (
        compute_intrinsic_angles(matrices0, common_line_direction),
        compute_intrinsic_angles(matrices1, common_line_direction),
    )
