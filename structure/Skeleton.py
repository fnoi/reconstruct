import itertools
import copy
import json
import os
import pickle

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import plotly.graph_objs as go
    from structure.Cloud import Segment
    from tools.geometry import warped_vectors_intersection, skew_lines
except ImportError as e:
    print(f'Error: {e}')


class Skeleton:
    def __init__(self, path: str, types: list, src=None, config=None):

        self.joint_frame = None
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

    def cache_pickle(self, path):
        with open(f'{path}/skeleton_cache.pickle', 'wb') as f:
            pickle.dump(self, f)

    def cache_json(self, path):
        export_dict = {}
        for i, bone in enumerate(self.bones):
            export_dict[f'bone_{i}'] = {
                'start': bone.line_cog_left.tolist(),
                'end': bone.line_cog_right.tolist(),
                'beam_verts': bone.h_beam_verts.tolist(),
                'rot_mat': bone.mat_rotation_xy.tolist(),
                'angle_xy': bone.angle_2D
            }
        with open(f'{path}/skeleton_cache.json', 'w') as f:
            json.dump(export_dict, f)


    def add_cloud(self, cloud):
        self.bones.append(cloud)
        with open(f'{self.path}/fresh_bone_{self.bone_count}.obj', 'w') as f:
            f.write(f'v {cloud.line_cog_left[0]} {cloud.line_cog_left[1]} {cloud.line_cog_left[2]} \n'
                    f'v {cloud.line_cog_right[0]} {cloud.line_cog_right[1]} {cloud.line_cog_right[2]} \n'
                    f'l 1 2 \n')
        self.bone_count += 1

    # def add_bone(self, bone):
    #     self.bones.append(bone)
    #     self.bone_count += 1

    def to_obj(self, topic: str, radius: bool = False):
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
        all_joints = list(itertools.combinations(range(self.bone_count), 2))
        self.joints_in = []
        for joint in all_joints:
            # calculate distance
            # here the error of not converging is "create"
            # bridgepoint1, bridgepoint2, rating, case, angle = warped_vectors_intersection(
            #     self.bones[joint[0]],
            #     self.bones[joint[1]])
            bridgepoint1, bridgepoint2, rating, case, angle = skew_lines(self.bones[joint[0]], self.bones[joint[1]])
            # print(rating)
            # if rating < self.threshold_distance_join:  ##
            self.joints_in.append([joint[0], joint[1], bridgepoint1, bridgepoint2, rating, case, angle])  # KEY

    def aggregate_bones(self):

        # self.plot_cog_skeleton() why here?

        while True:
            self.find_joints()
            self.joints2joint_array()
            self.joints2joint_frame()

            # if len(self.joints_array) == 19:
            #     a = 0

            # for-loop over rows in joints_array (which is a numpy array)
            for i, joint in enumerate(self.joints_array):
                id_bone_1 = int(joint[0])
                id_bone_2 = int(joint[1])
                bone_1 = self.bones[id_bone_1]
                bone_2 = self.bones[id_bone_2]
                acute_angle = min(joint[10], 180 - joint[10])

                # # for-loop over rows in joint_frame
                # for joint in self.joint_frame.iterrows():
                #     print(joint)
                #     bone_1 = self.bones[int(joint[1][0])]
                #     bone_2 = self.bones[int(joint[1][1])]
                #     acute_angle = min(joint[1]['angle'], 180 - joint[1]['angle'])
                if acute_angle < self.config.skeleton.aggregate_angle_max:
                    # find minimum distance
                    P0 = bone_1.line_cog_left
                    P1 = bone_1.line_cog_right
                    P2 = bone_2.line_cog_left
                    P3 = bone_2.line_cog_right
                    L0 = np.linalg.norm(P0 - P2)
                    L1 = np.linalg.norm(P0 - P3)
                    L2 = np.linalg.norm(P1 - P2)
                    L3 = np.linalg.norm(P1 - P3)
                    LX1 = np.linalg.norm(P0 - P1)
                    LX2 = np.linalg.norm(P2 - P3)

                    min_dist = min([L0, L1, L2, L3])
                    try:
                        dist_flag = min_dist > self.config.skeleton.aggregate_distance_max
                    except:
                        dist_flag = min_dist > 0.2

                    if dist_flag:
                        continue
                    else:
                        # identify main: longer
                        if LX1 >= LX2:
                            (ind_long, ind_short) = (id_bone_1, id_bone_2)
                        else:
                            (ind_long, ind_short) = (id_bone_2, id_bone_1)
                        points_long = self.bones[ind_long].points
                        points_short = self.bones[ind_short].points
                        print(f'currently {len(self.bones)} bones')

                        # remove both bones
                        if ind_long > ind_short:
                            segment_new = Segment(name=f'beam_{ind_short}', config=self.config)
                            segment_new.points = np.concatenate((points_long, points_short), axis=0)
                            self.bones.pop(ind_long)
                            self.bones.pop(ind_short)
                        else:
                            segment_new = Segment(name=f'beam_{ind_long}', config=self.config)
                            segment_new.points = np.concatenate((points_long, points_short), axis=0)
                            self.bones.pop(ind_short)
                            self.bones.pop(ind_long)

                        segment_new.calc_axes()
                        # add new bone
                        self.add_cloud(segment_new)
                        self.update_bones()
                        # self.add_bone(segment_new)
                        # try:
                        #     self.bones[-1].fit_cs_rev()
                        # except:
                        #     self.bones[-1].h_beam_params = False
                        #     self.bones[-1].h_beam_verts = False
                        # print(self.bones[-1].h_beam_params)

                        print(f'ummm now {len(self.bones)} bones')

                        break
                        # self.aggregate_bones()

            a = 0

            if i == self.joints_array.shape[0] - 1:
                self.plot_cog_skeleton()
                print('done with aggregation')
                break

                # redundant if recompute!
                # else:
                #     origin_coords = [-1e3, -1e3, -1e3]
                #     # identify main: longer
                #     if LX1 > LX2:
                #         long_ind = 0
                #         LA = L0
                #         # project P2 and P3 to 3D vector defined by P0 and P1
                #         P2 = P0 + np.dot(P2 - P0, P1 - P0) / np.dot(P1 - P0, P1 - P0) * (P1 - P0)
                #         P3 = P0 + np.dot(P3 - P0, P1 - P0) / np.dot(P1 - P0, P1 - P0) * (P1 - P0)
                #         # project origin_coords to line defined by P0 and P1
                #         origin = P0 + np.dot(origin_coords - P0, P1 - P0) / np.dot(P1 - P0, P1 - P0) * (P1 - P0)
                #         oP0 = np.linalg.norm(origin - P0)
                #         oP1 = np.linalg.norm(origin - P1)
                #         oP2 = np.linalg.norm(origin - P2)
                #         oP3 = np.linalg.norm(origin - P3)
                #
                #         if oP0 > oP1:
                #             P0_ = P1
                #             P1_ = P0
                #             P0 = P0_
                #             P1 = P1_
                #         if oP2 > oP3:
                #             P2_ = P3
                #             P3_ = P2
                #             P2 = P2_
                #             P3 = P3_
                #
                #         LB = np.linalg.norm(P0 - P2)
                #         LC = np.linalg.norm(P0 - P3)
                #
                #     else:
                #         long_ind = 1
                #         LA = L1
                #         # project P0 and P1 to 3D vector defined by P2 and P3
                #         P0 = P2 + np.dot(P0 - P2, P3 - P2) / np.dot(P3 - P2, P3 - P2) * (P3 - P2)
                #         P1 = P2 + np.dot(P1 - P2, P3 - P2) / np.dot(P3 - P2, P3 - P2) * (P3 - P2)
                #         # project origin_coords to line defined by P2 and P3
                #         origin = P2 + np.dot(origin_coords - P2, P3 - P2) / np.dot(P3 - P2, P3 - P2) * (P3 - P2)
                #         oP0 = np.linalg.norm(origin - P0)
                #         oP1 = np.linalg.norm(origin - P1)
                #         oP2 = np.linalg.norm(origin - P2)
                #         oP3 = np.linalg.norm(origin - P3)
                #
                #         if oP0 > oP1:
                #             P0_ = P1
                #             P1_ = P0
                #             P0 = P0_
                #             P1 = P1_
                #         if oP2 > oP3:
                #             P2_ = P3
                #             P3_ = P2
                #             P2 = P2_
                #             P3 = P3_
                #
                #         LB = np.linalg.norm(P0 - P2)
                #         LC = np.linalg.norm(P0 - P3)
                #
                #     # case A
                #     if LA <= LB:
                #         # in line, connect
                #         print('case A')
                #         # add small to long. re-do bone calc: endpoints will and direction might change!
                #
                #
                #         # remove small bone
                #     elif LB <= LA:
                #         # overlap, connect
                #         print('case B')
                #         # add small to long. re-do bone calc: endpoints and direction might change!
                #     elif LC <= LA:
                #         # integrate
                #         print('case C')
                #         # add small to long. re-do bone calc: endpoints (should) stay the same but recompute cant hurt
                #     else:
                #         raise Exception('case not covered')

                # case A: bones in line
                # remove smaller bone and add its points to the longer bone

                # TODO: after case is covered, recompute and re-start joint loop; recursion could be a smart move here, the bones and joint change in each iteration if one case ABC is covered
                # self.aggregate_bones()

                # # plot the two bone lines
                # fig = go.Figure()
                # fig.add_trace(go.Scatter3d(
                #     x=[P0[0], P1[0]],
                #     y=[P0[1], P1[1]],
                #     z=[P0[2], P1[2]],
                #     mode='lines',
                #     line=dict(
                #         color='red',
                #         width=6
                #     )
                # ))
                # fig.add_trace(go.Scatter3d(
                #     x=[P2[0], P3[0]],
                #     y=[P2[1], P3[1]],
                #     z=[P2[2], P3[2]],
                #     mode='lines',
                #     line=dict(
                #         color='blue',
                #         width=6
                #     )
                # ))
                # # start and endpoint of the longer bone
                # fig.add_trace(go.Scatter3d(
                #     x=[P0[0], P1[0]],
                #     y=[P0[1], P1[1]],
                #     z=[P0[2], P1[2]],
                #     mode='markers',
                #     marker=dict(
                #         size=10,
                #         color='red'
                #     )
                # ))
                # # start and endpoint of the shorter bone
                # fig.add_trace(go.Scatter3d(
                #     x=[P2[0], P3[0]],
                #     y=[P2[1], P3[1]],
                #     z=[P2[2], P3[2]],
                #     mode='markers',
                #     marker=dict(
                #         size=10,
                #         color='blue'
                #     )
                # ))
                # fig.show()

    def join_on_passing_v2(self):

        while True:

            self.find_joints()
            self.joints2joint_array()
            self.joints2joint_frame()

            agenda = self.joint_frame[(self.joint_frame['case'] == 0) | (self.joint_frame['case'] == 1)]
            # sort by rating
            agenda = agenda.sort_values(by='rating')
            # only rating < 0.3
            agenda = agenda[agenda['rating'] < 0.1]  #TODO: replace with config...

            edit_flag = False
            for joint in agenda.iterrows():
                if joint[1]['case'] == 0:  # bone_1 dominant
                    passing = int(joint[1]['bone1'])
                    joining = int(joint[1]['bone2'])
                    bridgepoint_joining = joint[1]['bridgepoint2']
                elif joint[1]['case'] == 1:  # bone_2 dominant
                    passing = int(joint[1]['bone2'])
                    joining = int(joint[1]['bone1'])
                    bridgepoint_joining = joint[1]['bridgepoint1']
                else:
                    raise Exception('joint type not covered')
                dist_left = np.linalg.norm(np.asarray(self.bones[joining].line_cog_left) - np.asarray(bridgepoint_joining))
                dist_right = np.linalg.norm(np.asarray(self.bones[joining].line_cog_right) - np.asarray(bridgepoint_joining))
                if dist_left < dist_right:
                    # if self.bones[joining].left_edit:  # edited before
                    #     continue
                    # else:
                    delta = np.linalg.norm(np.asarray(bridgepoint_joining) - np.asarray(self.bones[joining].line_cog_left))
                    if delta > 0:
                        self.bones[joining].line_cog_left = np.asarray(bridgepoint_joining)
                        # bone = self.bones[joining]
                        self.bones[joining].points_center = (np.asarray(self.bones[joining].line_cog_right) + np.asarray(self.bones[joining].line_cog_left)) / 2
                        self.bones[joining].left_edit = True
                        print(f'did sth, moved bone {joining} by {delta}')
                        edit_flag = True
                else:
                    # if self.bones[joining].right_edit:
                    #     continue
                    # else:
                    delta = np.linalg.norm(np.asarray(bridgepoint_joining) - np.asarray(self.bones[joining].line_cog_right))
                    if delta > 0:
                        self.bones[joining].line_cog_right = np.asarray(bridgepoint_joining)
                        # bone = self.bones[joining]
                        self.bones[joining].points_center = (np.asarray(self.bones[joining].line_cog_right) + np.asarray(self.bones[joining].line_cog_left)) / 2
                        self.bones[joining].right_edit = True
                        print(f'did sth, moved bone {joining} by {delta}')
                        edit_flag = True

                if edit_flag:
                    break  # break the for-loop over joints
                # self.bones[passing].intermediate_points.append(joint[1]['case'])

            if not edit_flag:
                break
        # for iter in self.bones:
        #     iter.left_edit = False
        #     iter.right_edit = False

        return




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
                    bone = self.bones[int(joining)]
                    bone.pca = (bone.line_raw_right - bone.line_raw_left) / np.linalg.norm(bone.line_raw_right - bone.line_raw_left)
                    bone.points_center = (bone.line_raw_right + bone.line_raw_left) / 2
                    self.bones[int(joining)].left_edit = True
                    print('did sth')
                    log += 1
            else:
                if self.bones[int(joining)].right_edit:
                    continue
                else:
                    self.bones[int(joining)].line_raw_right = np.array([joint[4], joint[5], joint[6]])
                    bone = self.bones[int(joining)]
                    bone.pca = (bone.line_raw_right - bone.line_raw_left) / np.linalg.norm(bone.line_raw_right - bone.line_raw_left)
                    bone.points_center = (bone.line_raw_right + bone.line_raw_left) / 2
                    self.bones[int(joining)].right_edit = True
                    print('did sth')
                    log += 1
            self.bones[int(passing)].intermediate_points.append(joint[2])

        for iter in self.bones:
            iter.left_edit = False
            iter.right_edit = False

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
                midpoint = np.array([joint[4], joint[5], joint[6]]) \
                           + (
                                   np.array([joint[4], joint[5], joint[6]])
                                   - np.array([joint[7], joint[8], joint[9]])) \
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

        while len(agenda) != 0:
            joints = []
            # remove tuppel where one side is already moved
            toremove = []
            for i in range(len(agenda)):

                toremove = []
                joints = [int(agenda[i][0]), int(agenda[i][1])]
                j = 1
                if i + j < len(agenda):
                    while agenda[i + j][0] == agenda[i][0]:
                        joints.append(int(agenda[i + j][1]))
                        j += 1
                        if i + j >= len(agenda):
                            break

                listdirection = []

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

                    dist_left = np.linalg.norm(self.bones[int(joints[iter + 1])].line_raw_left - np.array([agenda[i + iter][7], agenda[i + iter][8], agenda[i + iter][9]]))
                    dist_right = np.linalg.norm(self.bones[int(joints[iter + 1])].line_raw_right - np.array([agenda[i + iter][7], agenda[i + iter][8], agenda[i + iter][9]]))

                    # adding directions
                    if dist0_left < dist0_right:
                        if listdirection[0] == "right":
                            toremove.append(joints[iter + 1])
                            continue
                        else:
                            if dist_left < dist_right:
                                if self.bones[int(joints[iter + 1])].left_edit:
                                    toremove.append(joints[iter + 1])
                                    continue
                                listdirection.append("left")
                                self.bones[int(joints[iter + 1])].left_edit = True
                            else:
                                if self.bones[int(joints[iter + 1])].right_edit:
                                    toremove.append(joints[iter + 1])
                                    continue
                                listdirection.append("right")
                                self.bones[int(joints[iter + 1])].right_edit = True
                    else:
                        if listdirection[0] == "left":
                            toremove.append(joints[iter + 1])
                            continue
                        else:
                            if dist_left < dist_right:
                                if self.bones[int(joints[iter + 1])].left_edit:
                                    toremove.append(joints[iter + 1])
                                    continue
                                listdirection.append("left")
                                self.bones[int(joints[iter + 1])].left_edit = True
                            else:
                                if self.bones[int(joints[iter + 1])].right_edit:
                                    toremove.append(joints[iter + 1])
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
                    bone = self.bones[int(joints[iter])]
                    if listdirection[iter] == "left":
                        bone.line_raw_left = midpoint
                    else:
                        bone.line_raw_right = midpoint

                    # need to update the bones properties
                    bone.pca = (bone.line_raw_right - bone.line_raw_left) / np.linalg.norm(bone.line_raw_right - bone.line_raw_left)
                    bone.points_center = (bone.line_raw_right + bone.line_raw_left) / 2

                # set connection point as middel
                # join the most prominent
                for iter in np.unique(joints):
                    self.bones[int(iter)].intermediate_points.append(2)

            # reset edit that next iteration can modify them again
            for iter in self.bones:
                iter.left_edit = False
                iter.right_edit = False

            # re-calc joints and dists
            self.find_joints()
            self.joints2joint_array()
            agenda = self.joints_array[self.joints_array[:, 2] == 2]
            agenda = agenda[agenda[:, 0].argsort()]
            # remove all entrys where the bridge points are "identical"
            # need to add tolerance because the values are nearly identical
            agenda = [x for x in agenda if not np.allclose(np.array([x[4], x[5], x[6]]), np.array([x[7], x[8], x[9]]), rtol=1e-2)]

        # repeat until no more joints?
        if log == 0:
            self.potential[2] = 1
        if len(agenda) == 0:
            self.potential[2] = 1
        return

    def update_bones(self):
        for i, bone in enumerate(self.bones):
            bone.name = f'beam_{i}'
        self.bone_count = len(self.bones)
        return

    def joints2joint_array(self):
        joint_array = np.zeros((len(self.joints_in), 11))
        for i, joint in enumerate(self.joints_in):
            joint_array[i, 0] = int(joint[0])  # bone 1
            joint_array[i, 1] = int(joint[1])  # bone 2
            joint_array[i, 2] = int(joint[5])  # case
            joint_array[i, 3] = joint[4]  # rating

            joint_array[i, 4] = joint[2][0]  # bridgepoint1x
            joint_array[i, 5] = joint[2][1]  # bridgepoint1y
            joint_array[i, 6] = joint[2][2]  # bridgepoint1z

            joint_array[i, 7] = joint[3][0]  # bridgepoint2x
            joint_array[i, 8] = joint[3][1]  # bridgepoint2y
            joint_array[i, 9] = joint[3][2]  # bridgepoint2z

            joint_array[i, 10] = joint[6]  # angle

        self.joints_array = joint_array
        return

    def joints2joint_frame(self):
        joint_frame = pd.DataFrame(self.joints_array, columns=['bone1', 'bone2', 'case', 'rating', 'bp1x', 'bp1y', 'bp1z', 'bp2x', 'bp2y', 'bp2z', 'angle'])
        # collect bp1x, bp1y, bp1z, bp2x, bp2y, bp2z in one column
        joint_frame['bridgepoint1'] = joint_frame[['bp1x', 'bp1y', 'bp1z']].values.tolist()
        joint_frame['bridgepoint2'] = joint_frame[['bp2x', 'bp2y', 'bp2z']].values.tolist()
        joint_frame = joint_frame.drop(columns=['bp1x', 'bp1y', 'bp1z', 'bp2x', 'bp2y', 'bp2z'])

        # datatype conversion
        joint_frame['bone1'] = joint_frame['bone1'].astype(int)
        joint_frame['bone2'] = joint_frame['bone2'].astype(int)
        joint_frame['case'] = joint_frame['case'].astype(int)

        self.joint_frame = joint_frame
        return

    def plot_cog_skeleton(self, text=True, colorswitch=True, headline=None):
        # create plotly fig
        fig = go.Figure()

        num_bones = len(self.bones)

        if colorswitch:
            colorset = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            color_num = 10
            linecolor = []
            cogcolor = []
            pointcloudcolor = []

            for j in range(num_bones):
                color_index = j % color_num
                linecolor.append(colorset[color_index])
                cogcolor.append(colorset[color_index])
                pointcloudcolor.append(colorset[color_index])

        else:
            linecolor = ['blue' for _ in range(num_bones)]
            cogcolor = ['magenta' for _ in range(num_bones)]
            pointcloudcolor = ['grey' for _ in range(num_bones)]


        for i, bone in enumerate(self.bones):
            if bone.h_beam_params is False:
                continue
            # bone.cs_lookup()
            # bone.update_axes()
            # equal axis
            fig.update_layout(scene=dict(aspectmode='data'))

            # plot line_cog_left, line_cog_right as lines
            fig.add_trace(go.Scatter3d(x=[bone.line_cog_left[0], bone.line_cog_right[0]],
                                       y=[bone.line_cog_left[1], bone.line_cog_right[1]],
                                       z=[bone.line_cog_left[2], bone.line_cog_right[2]],
                                       mode='lines',
                                       line=dict(color=linecolor[i], width=3)))
            # add line_cog_left, line_cog_right as scatter points
            fig.add_trace(go.Scatter3d(x=[bone.line_cog_left[0], bone.line_cog_right[0]],
                                       y=[bone.line_cog_left[1], bone.line_cog_right[1]],
                                       z=[bone.line_cog_left[2], bone.line_cog_right[2]],
                                       mode='markers',
                                       marker=dict(color=cogcolor[i], size=5)))
            # point cloud scatter
            fig.add_trace(go.Scatter3d(x=bone.points[:, 0],
                                       y=bone.points[:, 1],
                                       z=bone.points[:, 2],
                                       mode='markers',
                                       marker=dict(color=pointcloudcolor[i], size=.6, opacity=0.8)))
            # add beam name to center point plus offset
            center_point = (bone.line_cog_right + bone.line_cog_left) / 2
            d_x = 0.4
            d_y = 0.3
            d_z = 0.2
            rel_dist = 0.1
            end_point = center_point + np.array([d_x, d_y, d_z])
            text_point = end_point - rel_dist * np.array([d_x, d_y, d_z])

            # add line from center_point to text_point
            if text:
                # split beam_name at _
                beam_name = bone.name.split('_')
                beam_no = int(beam_name[1])
                # add beam name to text point
                fig.add_trace(go.Scatter3d(x=[end_point[0]],
                                           y=[end_point[1]],
                                           z=[end_point[2]],
                                           mode='text',
                                           text=[beam_no],
                                           textposition='middle center',
                                           textfont=dict(family='Times New Roman', size=20, color='black')))

                fig.add_trace(go.Scatter3d(x=[center_point[0], text_point[0]],
                                            y=[center_point[1], text_point[1]],
                                            z=[center_point[2], text_point[2]],
                                            mode='lines',
                                            line=dict(color='black', width=2)))

        # perspective should be ortho
        fig.layout.scene.camera.projection.type = "orthographic"
        # no background grid
        fig.layout.scene.xaxis.visible = False
        fig.layout.scene.yaxis.visible = False
        fig.layout.scene.zaxis.visible = False

        elev = 30
        azim = -60
        r = 1.25  # Distance from center, you may need to adjust this
        x_eye = r * np.cos(np.radians(azim)) * np.cos(np.radians(elev))
        y_eye = r * np.sin(np.radians(azim)) * np.cos(np.radians(elev))
        z_eye = r * np.sin(np.radians(elev))

        fig.update_layout(scene_camera=dict(
            eye=dict(x=x_eye, y=y_eye, z=z_eye)
        ))

        fig.update_layout(
            scene=dict(
                aspectmode='data',
                # Optionally, set a custom aspect ratio
                # aspectratio=dict(x=1, y=1, z=1)
            )
        )

        if headline is not None:
            fig.update_layout(title_text=headline)

        # show go figure
        fig.show()
        return
