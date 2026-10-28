from __future__ import annotations

import numpy as np


def euler_zyz_to_matrix(
    rot: np.ndarray, tilt: np.ndarray, psi: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """
    Convert ZYZ Euler angles to rotation matrices. Input angles are given in
    radians as floating-point arrays and may have any broadcast-compatible shapes.

    Parameters
    ----------
    rot : np.ndarray
        First rotation angle about the z axis, in radians.
    tilt : np.ndarray
        Rotation angle about the y axis, in radians.
    psi : np.ndarray
        Final rotation angle about the z axis, in radians.
    out : np.ndarray, optional
        Array in which to store the resulting matrices. It must have
        shape ``broadcast_shape + (3, 3)`` and the same dtype as ``rot``.

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
