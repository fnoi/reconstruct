import numpy as np


def supernormal_svd(normals):
    U, S, Vt = np.linalg.svd(normals)
    return Vt[-1, :]


def supernormal_confidence(supernormal, normals):
    supernormal /= np.linalg.norm(supernormal)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    angles = np.arccos(np.dot(supernormal, normals.T))
    angles *= 180 / np.pi

    angles -= 90
    angles = np.abs(angles)

    return np.mean(angles)


def angular_deviation(vector, reference):
    vector /= np.linalg.norm(vector)
    reference /= np.linalg.norm(reference)
    angle = np.arccos(np.dot(vector, reference))
    return angle * 180 / np.pi
