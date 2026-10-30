from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EulerAngles:
    rot: np.ndarray | float
    tilt: np.ndarray | float
    psi: np.ndarray | float


def euler_zyz_to_matrix(
    angles: EulerAngles, out: np.ndarray | None = None
) -> np.ndarray:
    """
    Convert ZYZ Euler angles to rotation matrices. Input angles are given in
    radians as floating-point scalars or broadcast-compatible arrays.

    Parameters
    ----------
    angles : EulerAngles
        First z rotation (rot), intermediate y rotation (tilt), and final
        z rotation (psi), in radians.
    out : np.ndarray, optional
        Array in which to store the resulting matrices. It must have
        shape ``broadcast_shape + (3, 3)`` and the same dtype as ``angles.rot``.

    Returns
    -------
    np.ndarray
        Rotation matrices with shape ``broadcast_shape + (3, 3)``.

    Notes
    -----
    The matrix maps reference coordinates to image coordinates. In terms of
    standard active, right-handed rotations acting on column vectors, it is
    ``Rz(-psi) @ Ry(-tilt) @ Rz(-rot)``.

    The projection direction is given by the third row of the rotation matrix,
    ``euler_zyz_to_matrix(...)[..., 2, :]``.
    """
    rot = np.asarray(angles.rot)
    tilt = np.asarray(angles.tilt)
    psi = np.asarray(angles.psi)

    if any(not np.issubdtype(angle.dtype, np.floating) for angle in (rot, tilt, psi)):
        raise TypeError("Euler angle arrays must have real floating-point dtypes")

    # Create the output
    batch_shape = np.broadcast_shapes(rot.shape, tilt.shape, psi.shape)
    result_shape = batch_shape + (3, 3)
    dtype = rot.dtype

    if out is None:
        out = np.empty(result_shape, dtype=dtype)
    elif out.shape != result_shape or out.dtype != dtype:
        raise RuntimeError("Invalid output array was provided")

    ai = rot
    aj = tilt
    ak = psi

    # Obtain sin and cos of the angles
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)

    # Obtain the combinations
    cc = cj * ci
    cs = cj * si
    sc = sj * ci
    ss = sj * si

    # Build the matrix
    out[..., 0, 0] = ck * cc - sk * si
    out[..., 0, 1] = ck * cs + sk * ci
    out[..., 0, 2] = -ck * sj
    out[..., 1, 0] = -sk * cc - ck * si
    out[..., 1, 1] = -sk * cs + ck * ci
    out[..., 1, 2] = sk * sj
    out[..., 2, 0] = sc
    out[..., 2, 1] = ss
    out[..., 2, 2] = cj

    return out
