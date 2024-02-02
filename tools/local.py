import numpy as np


def supernormal_svd(normals):
    U, S, Vt = np.linalg.svd(normals)
    return Vt[-1, :]


def supernormal_confidence(supernormal, normals):
    """
    calculate confidence value of supernormal,
    confidence = md_sn / md_n
    md_sn: mean deviation angle between supernormal and normals (-90)
    md_n:  mean deviation angle between normals
    big sn, small n -> big confidence
    big sn, big n   -> small confidence
    small sn, big n -> very small confidence
    small sn, small n -> big confidence
    (is the idea, lets check)
    """
    supernormal /= np.linalg.norm(supernormal)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    angles = np.arccos(np.dot(supernormal, normals.T))
    angles = np.rad2deg(angles)
    # angles *= 180 / np.pi

    angles -= 90
    angles = np.abs(angles)

    md_sn = np.median(angles)

    mean_normal = np.mean(normals, axis=0)
    mean_normal /= np.linalg.norm(mean_normal)
    angles = np.arccos(np.dot(mean_normal, normals.T))
    angles = np.rad2deg(angles)
    # median
    md_n = np.median(angles)

    c = 0.1 * len(normals) * md_n / (0.5 * md_sn)

    return c


def angular_deviation(vector, reference):
    vector /= np.linalg.norm(vector)
    reference /= np.linalg.norm(reference)
    angle = np.arccos(np.dot(vector, reference))
    return angle * 180 / np.pi
