import itertools

import numpy as np

from tools.geometry import warped_vectors_intersection


class Skeleton:
    def __init__(self, path: str):
        self.joints = None
        self.path = path
        self.bones = []
        self.threshold_distance_join = 0.2

    def add(self, cloud):
        self.bones.append(cloud)

    def find_joints(self):
        all_joints = list(itertools.combinations(range(len(self.bones)), 2))
        self.joints_in = []
        for joint in all_joints:
            # calculate distance
            bridgepoint1, bridgepoint2, rating, case = warped_vectors_intersection(
                self.bones[joint[0]],
                self.bones[joint[1]])
            if rating < self.threshold_distance_join:
                self.joints_in.append({joint: (bridgepoint1, bridgepoint2, rating, case)})

    def join_on_passing(self):


    def join_dangling(self):


    def update_bones(self, cloud):