import numpy as np


def rotation_matrix_from_vectors(vec1, vec2):
    """Return matrix to rotate one vector to another.

    Parameters
    ----------
    vec1 : array-like
        Vector to rotate.
    vec2 : array-like
        Vector to rotate to.

    Returns
    -------
    R : array-like
        Rotation matrix.

    """

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    v = np.cross(vec1, vec2)
    s = np.linalg.norm(v)
    c = np.dot(vec1, vec2)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + np.dot(vx, vx) * (1 - c) / s ** 2
    return R
