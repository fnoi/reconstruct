import numpy as np


def supernormal_svd(normals):
    U, S, Vt = np.linalg.svd(normals)
    return Vt[-1, :]
