import itertools
import copy
import os

import numpy as np

from tools.geometry import warped_vectors_intersection


class Skeleton:
    def __init__(self, path: str, types: list):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        for in_type in types:
            if not os.path.exists(f'{self.path}/{in_type}'):
                os.makedirs(f'{self.path}/{in_type}')

        if 'beams' in types:
            self.beams = True
        if 'pipes' in types:
            self.pipes = True

        self.bones = []
        self.threshold_distance_join = 1
        self.bone_count = 0
        self.joints_in = None
        self.joints_array = None

    def add_cloud(self, cloud):
        self.bones.append(cloud)
        with open(f'{self.path}/fresh_bone_{self.bone_count}.obj', 'w') as f:
            f.write(f'v {cloud.left[0]} {cloud.left[1]} {cloud.left[2]} \n'
                    f'v {cloud.right[0]} {cloud.right[1]} {cloud.right[2]} \n'
                    f'l 1 2 \n')
        self.bone_count += 1

    def add_bone(self, bone):
        a = 0


    def to_obj(self, topic: str):
        for i, bone in enumerate(self.bones):
            with open(f'{self.path}/{topic}_bone_{i + 1}.obj', 'w') as f:
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
        # not sure if this is needed because case 0 will be handled by join_on_passing
        # passing = np.unique(
        #     np.hstack(
        #         (np.where(self.joints_array[:, 2] == 0), np.where(self.joints_array[:, 2] == 1))
        #     )
        # )
        agenda = self.joints_array[self.joints_array[:, 2] == 2]
        agenda = agenda[agenda[:, 0].argsort()]
        # reapeat the agenda until no more solution is found
        
        # 
        agenda_backup=[]
        while len(agenda)!=0:
            
            joints=[]
            # skip iterations in agenda if more points are on the same connections
            skipping=0
            # remove tuppel where one side is already moved
            toremove=[]
            for i in range(len(agenda)):
                    
                toremove=[]
                joints = [int(agenda[i][0]),int(agenda[i][1])]
                j=1
                if i<len(agenda)-1:
                    while agenda[i+j][0]==agenda[i][0]:
                        joints.append(int(agenda[i+j][1]))
                        j+=1
                listdirection=[]

                for iter in range(len(joints)-1):
                    # find right/left for both bones to calc midpoint

                    if iter==0:
                        dist_left = np.linalg.norm(self.bones[int(joints[iter])].left - np.array([agenda[i+iter][4], agenda[i+iter][5], agenda[i+iter][6]]))
                        dist_right = np.linalg.norm(self.bones[int(joints[iter])].right - np.array([agenda[i+iter][4], agenda[i+iter][5], agenda[i+iter][6]]))
                        if dist_left < dist_right:
                            if self.bones[int(joints[iter])].left_edit:
                                break
                            else:
                                listdirection.append("left")
                                self.bones[int(joints[iter])].left_edit = True
                        else:
                            if self.bones[int(joints[iter])].right_edit:
                                break
                            else:
                                listdirection.append("right")
                                self.bones[int(joints[iter])].right_edit = True
                    
                    # check if ther is a Z connection so every entry needs to be on the same side
                    dist0_left = np.linalg.norm(self.bones[int(joints[0])].left - np.array([agenda[i+iter][4], agenda[i+iter][5], agenda[i+iter][6]]))
                    dist0_right = np.linalg.norm(self.bones[int(joints[0])].right - np.array([agenda[i+iter][4], agenda[i+iter][5], agenda[i+iter][6]]))

                    dist_left = np.linalg.norm(self.bones[int(joints[iter+1])].left - np.array([agenda[i+iter][7], agenda[i+iter][8], agenda[i+iter][9]]))
                    dist_right = np.linalg.norm(self.bones[int(joints[iter+1])].right - np.array([agenda[i+iter][7], agenda[i+iter][8], agenda[i+iter][9]]))
                    
                    if dist0_left < dist0_right:
                        if listdirection[0]=="right" or self.bones[int(joints[iter+1])].left_edit:
                            if listdirection[0]=="right":
                                toremove.append(joints[iter])
                            continue
                        else:
                            if dist_left<dist_right:
                                listdirection.append("left")
                                self.bones[int(joints[iter+1])].left_edit = True
                            else:
                                listdirection.append("right")
                                self.bones[int(joints[iter+1])].right_edit = True
                    else:
                        if listdirection[0]=="left" or self.bones[int(joints[iter+1])].right_edit:
                            if listdirection[0]=="left":
                                toremove.append(joints[iter+1])
                            continue
                        else:
                            if dist_left<dist_right:
                                listdirection.append("left")
                                self.bones[int(joints[iter+1])].left_edit = True
                            else:
                                listdirection.append("right")
                                self.bones[int(joints[iter+1])].right_edit = True
                    
                    
                # if no directions // only one no need to iterate
                if not listdirection or len(listdirection)==1:
                    continue
                
                if toremove:
                    for m in toremove:
                        joints.remove(m)
                
                # this should not happen
                if len(listdirection)!=len(joints):
                    a=0
                # calc mid point
                # find all bridgepoints to calc the best midpoint for all endpoints
                midpoint=np.asarray([0,0,0])
                tosearch=list(itertools.combinations(np.unique(joints), 2))
                count=0
                #this can be made faster with np
                for iter in tosearch:
                    for m in range(len(agenda)):
                        if iter[0]==agenda[m][0] and iter[1]==agenda[m][1] or iter[1]==agenda[m][0] and iter[0]==agenda[m][1]:
                            tmp=agenda[m]
                            midpoint=midpoint+np.asarray([tmp[4],tmp[5],tmp[6]])+np.asarray([tmp[7],tmp[8],tmp[9]])
                            count=count+1
                            break

                # calc mean of midpoints 
                # *2 because of the mean of midpoint
                midpoint=midpoint/(count*2)
                for iter in range(len(joints)):
                    if listdirection[iter]=="left":
                        self.bones[int(joints[iter])].left=midpoint
                    else:
                        self.bones[int(joints[iter])].right=midpoint
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
            # need to add tolerance because the values are nearly identical but it seems that some calc are in between
            agenda = [x for x in agenda if not np.allclose(np.array([x[4], x[5], x[6]]),np.array([x[7], x[8], x[9]]),rtol=1e-2)]
            # if the solution is not convercing -> break
            if len(agenda)==len(agenda_backup):
                if np.allclose(agenda,agenda_backup):
                    return
            agenda_backup=copy.deepcopy(agenda)
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
