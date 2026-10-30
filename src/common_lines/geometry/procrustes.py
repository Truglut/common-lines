import numpy as np


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    """
    Return the rotation matrix closest to a 3x3 matrix in Frobenius norm.

    The returned matrix R solves

        minimize ||matrix - R||_F
        subject to R.T @ R = I and det(R) = 1.

    Parameters
    ----------
    matrix
        Real-valued array with shape (3, 3).

    Returns
    -------
    numpy.ndarray
        The closest rotation matrix, with shape (3, 3).

    Raises
    ------
    ValueError
        If the input does not have shape (3, 3), contains non-finite
        values, or is complex-valued.
    """
    matrix = np.asarray(matrix)

    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a matrix with shape (3, 3), got {matrix.shape}.")

    if np.iscomplexobj(matrix):
        raise ValueError("The input matrix must be real-valued.")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("The input matrix must contain only finite values.")

    # Convert integer and low-precision inputs before computing the SVD.
    matrix = np.asarray(matrix, dtype=np.float64)

    u, _, vt = np.linalg.svd(matrix)

    # U @ Vt is the nearest orthogonal matrix. When its determinant is
    # negative, flip the singular-vector direction associated with the
    # smallest singular value to obtain a proper rotation.
    correction = np.ones(3)
    correction[-1] = np.linalg.det(u @ vt)

    return (u * correction) @ vt
