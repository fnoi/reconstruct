import itertools

import numpy as np

from tools.geometry import warped_vectors_intersection


class Skeleton:
    def __init__(self, path: str):
        self.joints = None
        self.path = path
        self.bones = []
        self.threshold_distance_join = 1
        self.bone_count = 0
        self.joints_in = None
        self.joints_array = None


    def add(self, cloud):
        self.bones.append(cloud)
        with open(f'{self.path}/fresh_bone_{self.bone_count}.obj', 'w') as f:
            f.write(f'v {cloud.left[0]} {cloud.left[1]} {cloud.left[2]} \n'
                    f'v {cloud.right[0]} {cloud.right[1]} {cloud.right[2]} \n'
                    f'l 1 2 \n')
        self.bone_count += 1

    def to_obj(self, topic: str):
        for i, bone in enumerate(self.bones):
            with open(f'{self.path}/{topic}_bone_{i}.obj', 'w') as f:
                f.write(f'v {bone.left[0]} {bone.left[1]} {bone.left[2]} \n'
                        f'v {bone.right[0]} {bone.right[1]} {bone.right[2]} \n'
                        f'l 1 2 \n')

    def find_joints(self):
        all_joints = list(itertools.combinations(range(len(self.bones)), 2))
        self.joints_in = []
        for joint in all_joints:
            # calculate distance
            bridgepoint1, bridgepoint2, rating, case = warped_vectors_intersection(
                self.bones[joint[0]],
                self.bones[joint[1]])
            # print(rating)
            if rating < self.threshold_distance_join:
                self.joints_in.append([joint[0], joint[1], bridgepoint1, bridgepoint2, rating, case])  # KEY

    def join_on_passing(self):
        self.joints2joint_array()
        agenda = self.joints_array[self.joints_array[:, 2] == 0]
        agenda = agenda[agenda[:, 3].argsort()]
        for joint in agenda:
            passing = joint[0]
            joining = joint[1]
            dist_left = np.linalg.norm(self.bones[int(joining)].left - np.array([joint[4], joint[5], joint[6]]))
            dist_right = np.linalg.norm(self.bones[int(joining)].right - np.array([joint[4], joint[5], joint[6]]))
            if dist_left < dist_right:
                if self.bones[int(joining)].left_edit:
                    continue
                else:
                    self.bones[int(joining)].left = np.array([joint[4], joint[5], joint[6]])
                    self.bones[int(joining)].left_edit = True
            else:
                if self.bones[int(joining)].right_edit:
                    continue
                else:
                    self.bones[int(joining)].right = np.array([joint[4], joint[5], joint[6]])
                    self.bones[int(joining)].right_edit = True
            self.bones[int(passing)].intermediate_points.append(joint[2])

        return

    def join_passing(self):
        # find passing bones #TODO: implementation should pick up where we left off here
        self.joints2joint_array()
        passing = np.unique(
            np.hstack(
                (np.where(self.joints_array[:, 2] == 0), np.where(self.joints_array[:, 2] == 1))
            ), return_counts=True
        )


        a = 0
        # join the most prominent
        # re-calc joints and dists
        # repeat until no more joints?
        return

    def update_bones(self, cloud):
        return

    def joints2joint_array(self):
        joint_array = np.zeros((len(self.joints_in), 10))
        for i, joint in enumerate(self.joints_in):
            joint_array[i, 0] = joint[0]  # bone 1
            joint_array[i, 1] = joint[1]  # bone 2
            joint_array[i, 2] = joint[5]  # case
            joint_array[i, 3] = joint[4]  # rating

            joint_array[i, 4] = joint[2][0]  # bridgepoint1x
            joint_array[i, 5] = joint[2][1]  # bridgepoint1y
            joint_array[i, 6] = joint[2][2]  # bridgepoint1z

            joint_array[i, 7] = joint[3][0]  # bridgepoint2x
            joint_array[i, 8] = joint[3][1]  # bridgepoint2y
            joint_array[i, 9] = joint[3][2]  # bridgepoint2z
        self.joints_array = joint_array
        return
