import numpy as np


class PointCloud(object):
    def __init__(self, points: np.ndarray, normals=None, colors=None, features=None):
        self.points = points
        self.normals = normals
        self.colors = colors
        self.features = features


