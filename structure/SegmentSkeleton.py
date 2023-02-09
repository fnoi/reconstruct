import itertools

import numpy as np

from tools.geometry import warped_vectors_intersection


class Skeleton:
    def __init__(self, path: str):
        self.joints = None
        self.path = path
        self.bones = []
        self.threshold_distance_join = 0.2
        self.bone_count = 0

    def add(self, cloud):
        self.bones.append(cloud)
        with open(f'{self.path}/fresh_bone_{self.bone_count}.obj', 'w') as f:
            f.write(f'v {cloud.left[0]} {cloud.left[1]} {cloud.left[2]} \n'
                    f'v {cloud.right[0]} {cloud.right[1]} {cloud.right[2]} \n'
                    f'l 1 2 \n')
        self.bone_count += 1

    def find_joints(self):
        all_joints = list(itertools.combinations(range(len(self.bones)), 2))
        self.joints_in = []
        for joint in all_joints:
            # calculate distance
            bridgepoint1, bridgepoint2, rating, case = warped_vectors_intersection(
                self.bones[joint[0]],
                self.bones[joint[1]])
            if rating < self.threshold_distance_join:
                self.joints_in.append([joint[0], joint[1], bridgepoint1, bridgepoint2, rating, case])  # KEY

    def join_on_passing(self):
        joint_array = np.zeros((len(self.joints_in), 4))
        for i, joint in enumerate(self.joints_in):
            joint_array[i, 0] = joint[0]  # bone 1
            joint_array[i, 1] = joint[1]  # bone 2
            joint_array[i, 3] = joint[4]  # rating
            joint_array[i, 2] = joint[5]  # case
        agenda = joint_array[np.where(joint_array[:, 3] == 0)]

        ratings = np.zeros((len(self.joints_in), 1))
        for i, joint in enumerate(self.joints_in):
            if joint[-1] == 0:
                ratings[i] = joint[-2]
        a = 0

        return

    def join_dangling(self):
        return

    def update_bones(self, cloud):
        return
