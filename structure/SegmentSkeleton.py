import itertools
import copy
import os

import numpy as np

from tools.geometry import warped_vectors_intersection


class Skeleton:
    def __init__(self, path: str, types: list, src=None, config=None):

        self.potential = None
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        for in_type in types:
            if not os.path.exists(f'{self.path}/{in_type}'):
                os.makedirs(f'{self.path}/{in_type}')

        self.beams = False
        self.pipes = False

        if 'beams' in types:
            self.beams = True
        if 'pipes' in types:
            self.pipes = True

        self.bones = []
        self.threshold_distance_join = 1
        self.threshold_distance_trim = 0.4
        self.bone_count = 0
        self.joints_in = None
        self.joints_array = None

        self.config = config  # logic from dataframe revision

    def add_cloud(self, cloud):
        self.bones.append(cloud)
        with open(f'{self.path}/fresh_bone_{self.bone_count}.obj', 'w') as f:
            f.write(f'v {cloud.line_raw_left[0]} {cloud.line_raw_left[1]} {cloud.line_raw_left[2]} \n'
                    f'v {cloud.line_raw_right[0]} {cloud.line_raw_right[1]} {cloud.line_raw_right[2]} \n'
                    f'l 1 2 \n')
        self.bone_count += 1

    def add_bone(self, bone):
        self.bones.append(bone)
        self.bone_count += 1

    def to_obj(self, topic: str, radius: bool=False):
        for i, bone in enumerate(self.bones):
            with open(f'{self.path}/{topic}_bone_{i + 1}.obj', 'w') as f:
                if radius:
                    f.write(f'v {bone.line_raw_left[0]} {bone.line_raw_left[1]} {bone.line_raw_left[2]} \n'
                            f'v {bone.line_raw_right[0]} {bone.line_raw_right[1]} {bone.line_raw_right[2]} \n'
                            f'l 1 2 \n'
                            f'# radius: {bone.radius} \n')
                else:
                    f.write(f'v {bone.line_raw_left[0]} {bone.line_raw_left[1]} {bone.line_raw_left[2]} \n'
                            f'v {bone.line_raw_right[0]} {bone.line_raw_right[1]} {bone.line_raw_right[2]} \n'
                            f'l 1 2 \n')

    def find_joints(self):
        all_joints = list(itertools.combinations(range(len(self.bones)), 2))
        self.joints_in = []
        for joint in all_joints:

            # calculate distance
            # here the error of not converging is "create"
            bridgepoint1, bridgepoint2, rating, case = warped_vectors_intersection(
                self.bones[joint[0]],
                self.bones[joint[1]])
            # print(rating)
            if rating < self.threshold_distance_join:
                self.joints_in.append([joint[0], joint[1], bridgepoint1, bridgepoint2, rating, case])  # KEY

    def join_on_passing(self):
        log = 0
        self.find_joints()
        self.joints2joint_array()
        agenda = self.joints_array[self.joints_array[:, 2] == 0]
        agenda = agenda[agenda[:, 3].argsort()]
        for joint in agenda:
            passing = joint[0]
            joining = joint[1]
            dist_left = np.linalg.norm(self.bones[int(joining)].line_raw_left - np.array([joint[4], joint[5], joint[6]]))
            dist_right = np.linalg.norm(self.bones[int(joining)].line_raw_right - np.array([joint[4], joint[5], joint[6]]))
            if dist_left < dist_right:
                if self.bones[int(joining)].left_edit:
                    continue
                else:
                    self.bones[int(joining)].line_raw_left = np.array([joint[4], joint[5], joint[6]])
                    bone=self.bones[int(joining)]
                    bone.pca= (bone.line_raw_right - bone.line_raw_left) / np.linalg.norm(bone.line_raw_right - bone.line_raw_left)
                    bone.points_center= (bone.line_raw_right + bone.line_raw_left) / 2
                    self.bones[int(joining)].left_edit = True
                    print('did sth')
                    log += 1
            else:
                if self.bones[int(joining)].right_edit:
                    continue
                else:
                    self.bones[int(joining)].line_raw_right = np.array([joint[4], joint[5], joint[6]])
                    bone=self.bones[int(joining)]
                    bone.pca= (bone.line_raw_right - bone.line_raw_left) / np.linalg.norm(bone.line_raw_right - bone.line_raw_left)
                    bone.points_center= (bone.line_raw_right + bone.line_raw_left) / 2
                    self.bones[int(joining)].right_edit = True
                    print('did sth')
                    log += 1
            self.bones[int(passing)].intermediate_points.append(joint[2])


        for iter in self.bones:
            iter.left_edit=False
            iter.right_edit=False

        self.find_joints()
        self.joints2joint_array()
        agenda = self.joints_array[self.joints_array[:, 2] == 1]
        agenda = agenda[agenda[:, 3].argsort()]
        for joint in agenda:
            passing = joint[1]
            joining = joint[0]
            dist_left = np.linalg.norm(self.bones[int(joining)].line_raw_left - np.array([joint[7], joint[8], joint[9]]))
            dist_right = np.linalg.norm(self.bones[int(joining)].line_raw_right - np.array([joint[7], joint[8], joint[9]]))
            if dist_left < dist_right:
                if self.bones[int(joining)].left_edit:
                    continue
                else:
                    self.bones[int(joining)].line_raw_left = np.array([joint[7], joint[8], joint[9]])
                    self.bones[int(joining)].left_edit = True
            else:
                if self.bones[int(joining)].right_edit:
                    continue
                else:
                    self.bones[int(joining)].line_raw_right = np.array([joint[7], joint[8], joint[9]])
                    self.bones[int(joining)].right_edit = True
            self.bones[int(passing)].intermediate_points.append(joint[2])

        if log == 0:
            self.potential[0] = 1
        return

    def trim_passing(self):
        log = 0
        self.find_joints()
        self.joints2joint_array()
        agenda = self.joints_array[self.joints_array[:, 2] == 3]
        agenda = agenda[agenda[:, 3].argsort()]
        for joint in agenda:
            # trim both
            protrusions_1 = np.array([
                np.linalg.norm(self.bones[int(joint[0])].line_raw_left - np.array([joint[4], joint[5], joint[6]])),
                np.linalg.norm(self.bones[int(joint[0])].line_raw_right - np.array([joint[4], joint[5], joint[6]]))
            ])
            case_1 = np.argmin(protrusions_1)
            protrusions_2 = np.array([
                np.linalg.norm(self.bones[int(joint[1])].line_raw_left - np.array([joint[4], joint[5], joint[6]])),
                np.linalg.norm(self.bones[int(joint[1])].line_raw_right - np.array([joint[4], joint[5], joint[6]]))
            ])
            case_2 = np.argmin(protrusions_2)
            if protrusions_1[case_1] < self.threshold_distance_trim and protrusions_2[
                case_2] < self.threshold_distance_trim:
                midpoint = np.array([joint[4], joint[5], joint[6]]) + np.array(
                    np.array([joint[7], joint[8], joint[9]]) - np.array([joint[4], joint[5], joint[6]])
                ) / 2

                if case_1 == 0:
                    self.bones[int(joint[0])].line_raw_left = midpoint
                    self.bones[int(joint[0])].left_edit = True
                    print('did sth')
                    log += 1
                else:
                    self.bones[int(joint[0])].line_raw_right = midpoint
                    self.bones[int(joint[0])].right_edit = True
                    print('did sth')
                    log += 1
                if case_2 == 0:
                    self.bones[int(joint[1])].line_raw_left = midpoint
                    self.bones[int(joint[1])].left_edit = True
                    print('did sth')
                    log += 1
                else:
                    self.bones[int(joint[1])].line_raw_right = midpoint
                    self.bones[int(joint[1])].right_edit = True
                    print('did sth')
                    log += 1

            else:


                # trim one
                for i in [0, 1]:
                    bone = self.bones[i]
                    protrusion_left = np.linalg.norm(bone.line_raw_left - np.array([joint[4], joint[5], joint[6]]))
                    protrusion_right = np.linalg.norm(bone.line_raw_right - np.array([joint[4], joint[5], joint[6]]))

                    if protrusion_left < protrusion_right and protrusion_left < self.threshold_distance_trim and joint[3] < self.threshold_distance_join and not bone.left_edit:
                        if i == 0:
                            bone.line_raw_left = np.array([joint[7], joint[8], joint[9]])
                        elif i == 1:
                            bone.line_raw_left = np.array([joint[4], joint[5], joint[6]])
                        else:
                            raise Exception('bone index out of range')
                        bone.left_edit = True
                        print('did sth')
                        log += 1

                    if protrusion_right < protrusion_left and protrusion_right < self.threshold_distance_trim and joint[3] < self.threshold_distance_join and not bone.right_edit:
                        if i == 0:
                            bone.line_raw_right = np.array([joint[7], joint[8], joint[9]])
                        elif i == 1:
                            bone.line_raw_right = np.array([joint[4], joint[5], joint[6]])
                        else:
                            raise Exception('bone index out of range')
                        bone.right_edit = True
                        print('did sth')
                        log += 1
                    if log == 2:
                        bone.left_joint = True
                        bone.right_joint = True
        if log == 0:
            self.potential[1] = 1
        return

    def join_passing_new(self):  #TODO: should start with only real passings... and finally join non-passings
        log = 0
        agenda = [1]
        count = 0
        while len(agenda) > 0:
            count += 1
            self.find_joints()
            self.joints2joint_array()
            agenda = self.joints_array[self.joints_array[:, 2] == 2]
            agenda = agenda[agenda[:, 3].argsort()]
            for joint in agenda:

                # what are the relevant ends?
                midpoint = np.array([joint[4], joint[5], joint[6]])\
                           + (
                                   np.array([joint[4], joint[5], joint[6]])
                                   - np.array([joint[7], joint[8], joint[9]]))\
                           / 2

                dists_1 = np.array([
                    np.linalg.norm(self.bones[int(joint[0])].line_raw_left - midpoint),
                    np.linalg.norm(self.bones[int(joint[0])].line_raw_right - midpoint)
                ])
                case_1 = np.argmin(dists_1)  # 0 = left, 1 = right
                dists_2 = np.array([
                    np.linalg.norm(self.bones[int(joint[1])].line_raw_left - midpoint),
                    np.linalg.norm(self.bones[int(joint[1])].line_raw_right - midpoint)
                ])
                case_2 = np.argmin(dists_2)  # 0 = left, 1 = right

                if dists_1[case_1] < self.threshold_distance_trim and dists_2[case_2] < self.threshold_distance_trim:
                    if case_1 == 0:
                        if self.bones[int(joint[0])].left_edit:
                            continue
                        self.bones[int(joint[0])].line_raw_left = midpoint
                        self.bones[int(joint[0])].left_edit = True
                        print('did sth')
                    else:
                        if self.bones[int(joint[0])].right_edit:
                            continue
                        self.bones[int(joint[0])].line_raw_right = midpoint
                        self.bones[int(joint[0])].right_edit = True
                        print('did sth')
                    if case_2 == 0:
                        if self.bones[int(joint[1])].left_edit:
                            continue
                        self.bones[int(joint[1])].line_raw_left = midpoint
                        self.bones[int(joint[1])].left_edit = True
                        print('did sth')
                    else:
                        if self.bones[int(joint[1])].right_edit:
                            continue
                        self.bones[int(joint[1])].line_raw_right = midpoint
                        self.bones[int(joint[1])].right_edit = True
                        print('did sth')
                    log += 1
                    break
            if count > 5:
                break
            if log == 0:
                self.potential[0] = 1
                break
        return

    def join_passing(self):
        self.find_joints()
        log = 0
        # find passing bones #TODO: implementation should pick up where we left off here
        self.joints2joint_array()
        agenda = self.joints_array[self.joints_array[:, 2] == 2]
        agenda = agenda[agenda[:, 0].argsort()]
        # reapeat the agenda until no more solution is found
        
        while len(agenda)!=0:
            joints=[]
            # remove tuppel where one side is already moved
            toremove = []
            for i in range(len(agenda)):

                toremove=[]
                joints = [int(agenda[i][0]),int(agenda[i][1])]
                j=1
                if i+j<len(agenda):
                    while agenda[i+j][0]==agenda[i][0]:
                        joints.append(int(agenda[i+j][1]))
                        j+=1
                        if i+j>=len(agenda):
                            break

                listdirection=[]

                for iter in range(len(joints) - 1):
                    # find right/left for both bones to calc midpoint

                    if iter == 0:
                        dist_left = np.linalg.norm(self.bones[int(joints[iter])].line_raw_left - np.array(
                            [agenda[i + iter][4], agenda[i + iter][5], agenda[i + iter][6]]))
                        dist_right = np.linalg.norm(self.bones[int(joints[iter])].line_raw_right - np.array(
                            [agenda[i + iter][4], agenda[i + iter][5], agenda[i + iter][6]]))
                        if dist_left < dist_right:
                            if self.bones[int(joints[iter])].left_edit:
                                break
                            else:
                                listdirection.append("left")
                                self.bones[int(joints[iter])].left_edit = True
                                log += 1
                        else:
                            if self.bones[int(joints[iter])].right_edit:
                                break
                            else:
                                listdirection.append("right")
                                self.bones[int(joints[iter])].right_edit = True
                                log += 1

                    # check if ther is a Z connection so every entry needs to be on the same side
                    dist0_left = np.linalg.norm(self.bones[int(joints[0])].line_raw_left - np.array(
                        [agenda[i + iter][4], agenda[i + iter][5], agenda[i + iter][6]]))
                    dist0_right = np.linalg.norm(self.bones[int(joints[0])].line_raw_right - np.array(
                        [agenda[i + iter][4], agenda[i + iter][5], agenda[i + iter][6]]))

                    dist_left = np.linalg.norm(self.bones[int(joints[iter+1])].line_raw_left - np.array([agenda[i + iter][7], agenda[i + iter][8], agenda[i + iter][9]]))
                    dist_right = np.linalg.norm(self.bones[int(joints[iter+1])].line_raw_right - np.array([agenda[i + iter][7], agenda[i + iter][8], agenda[i + iter][9]]))

                    # adding directions
                    if dist0_left < dist0_right:
                        if listdirection[0]=="right":
                            toremove.append(joints[iter+1])
                            continue
                        else:
                            if dist_left<dist_right:
                                if self.bones[int(joints[iter+1])].left_edit:
                                    toremove.append(joints[iter+1])
                                    continue
                                listdirection.append("left")
                                self.bones[int(joints[iter + 1])].left_edit = True
                            else:
                                if self.bones[int(joints[iter+1])].right_edit:
                                    toremove.append(joints[iter+1])
                                    continue
                                listdirection.append("right")
                                self.bones[int(joints[iter + 1])].right_edit = True
                    else:
                        if listdirection[0]=="left":
                            toremove.append(joints[iter+1])
                            continue
                        else:
                            if dist_left<dist_right:
                                if self.bones[int(joints[iter+1])].left_edit:
                                    toremove.append(joints[iter+1])
                                    continue
                                listdirection.append("left")
                                self.bones[int(joints[iter + 1])].left_edit = True
                            else:
                                if self.bones[int(joints[iter+1])].right_edit:
                                    toremove.append(joints[iter+1])
                                    continue
                                listdirection.append("right")
                                self.bones[int(joints[iter + 1])].right_edit = True

                # if no directions // only one no need to iterate
                if not listdirection or len(listdirection) == 1:
                    continue

                if toremove:
                    for m in toremove:
                        joints.remove(m)

                # this should not happen
                if len(listdirection) != len(joints):
                    a = 0
                # calc mid point
                # find all bridgepoints to calc the best midpoint for all endpoints
                midpoint = np.asarray([0, 0, 0])
                tosearch = list(itertools.combinations(np.unique(joints), 2))
                count = 0
                # this can be made faster with np
                for iter in tosearch:
                    for m in range(len(agenda)):
                        if iter[0] == agenda[m][0] and iter[1] == agenda[m][1] or iter[1] == agenda[m][0] and iter[0] == \
                                agenda[m][1]:
                            tmp = agenda[m]
                            midpoint = midpoint + np.asarray([tmp[4], tmp[5], tmp[6]]) + np.asarray(
                                [tmp[7], tmp[8], tmp[9]])
                            count = count + 1
                            break

                # calc mean of midpoints 
                # *2 because of the mean of midpoint
                midpoint = midpoint / (count * 2)
                for iter in range(len(joints)):
                    bone=self.bones[int(joints[iter])]
                    if listdirection[iter]=="left":
                        bone.line_raw_left=midpoint
                    else:
                        bone.line_raw_right=midpoint

                    # need to update the bones properties
                    bone.pca= (bone.line_raw_right - bone.line_raw_left) / np.linalg.norm(bone.line_raw_right - bone.line_raw_left)
                    bone.points_center= (bone.line_raw_right + bone.line_raw_left) / 2

                # set connection point as middel
                # join the most prominent
                for iter in np.unique(joints):
                    self.bones[int(iter)].intermediate_points.append(2)

            # reset edit that next iteration can modify them again
            for iter in self.bones:
                iter.left_edit=False
                iter.right_edit=False



            # re-calc joints and dists
            self.find_joints()
            self.joints2joint_array()
            agenda = self.joints_array[self.joints_array[:, 2] == 2]
            agenda = agenda[agenda[:, 0].argsort()]
            # remove all entrys where the bridge points are "identical"
            # need to add tolerance because the values are nearly identical
            agenda = [x for x in agenda if not np.allclose(np.array([x[4], x[5], x[6]]),np.array([x[7], x[8], x[9]]),rtol=1e-2)]

        # repeat until no more joints?
        if log == 0:
            self.potential[2] = 1
        if len(agenda) == 0:
            self.potential[2] = 1
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
