import itertools
import os

import numpy as np

from structure.CloudSegment import CloudSegment
from structure.SegmentSkeleton import Skeleton
from tools.geometry import vector_distance, manipulate_skeleton
from tools.IO import lines2obj
from tools.utils import update_logbook_checklist

if __name__ == '__main__':

    skeleton = Skeleton(f'{str(os.getcwd())}/data/out/0_skeleton')
    if not os.path.exists(skeleton.path):
        os.makedirs(skeleton.path)

    segments: list = [
        'beam_01',
        'beam_02',
        'beam_03',
        'beam_04',
        'beam_05',
        'beam_06',
        'beam_07',
        'beam_08',
        'beam_09',
        'beam_10'
    ]

    skeleton = []
    for segment in segments:
        cloud = CloudSegment(name=segment)
        cloud.load_from_txt(segment)
        cloud.calc_pca_o3d()
        cloud.transform()
        skeleton.append(cloud)

    # build the initial skeleton
    for bone in skeleton:
        lines2obj([(bone.left, bone.right)], path=skeleton.path, topic=f'raw_{bone.name}')

    # logbook: list of tuples (bone, bone, distance, case) to describe state of skeleton joints
    logbook = {}
    # done_or_flagged: list of indices of bones that are done or flagged (legacy?)
    done_or_flagged = []
    # checklist: dictionary indicating which bones are passing through (prio 1)
    checklist = dict.fromkeys(range(len(skeleton)), 0)
    # neighbors: list of tuples (bone, bone) to describe all potential bone joints
    neighbors = list(itertools.combinations(range(len(skeleton)), 2))

    while True:
        min_ind = None
        min_track = 1e10
        logbook, checklist = update_logbook_checklist(skeleton=skeleton,
                                                      neighbors=neighbors,
                                                      checklist=checklist)

        ind = max(checklist, key=checklist.get)
        print(ind)
        while checklist[ind] > 0:

            ind = max(checklist, key=checklist.get)
            temp_log = []
            for neighbor in neighbors:
                if ind in neighbor:
                    temp_log.append(logbook[neighbor][2])
                else:
                    temp_log.append(1e10)
            to_do = np.argmin(np.asarray(temp_log))
            skeleton[neighbors[to_do][0]], skeleton[neighbors[to_do][1]] = manipulate_skeleton(
                segment1=skeleton[neighbors[to_do][0]],
                segment2=skeleton[neighbors[to_do][1]],
                bridgepoint1=logbook[neighbors[to_do]][0],
                bridgepoint2=logbook[neighbors[to_do]][1],
                case=logbook[neighbors[to_do]][3])
            # initialize next iteration
            done_or_flagged.append(neighbors[to_do])
            checklist[ind] -= 1
            ind = max(checklist, key=checklist.get)


        if checklist[neighbors[to_do][0]] == 2:
            if rating < min_track and neighbor not in done_or_flagged:
                min_track = rating
                min_ind = neighbor
                min_bp1 = bridgepoint1
                min_bp2 = bridgepoint2
                min_case = case
            logbook[neighbor] = rating
            # print(dist)
            # lines2obj([(pt1, pt2)], path=skeletonpath, topic=f'cross_me_{str(neighbor)}')

        boss = max(checklist, key=checklist.get)

        if min_ind is None or min_track > 0.3:
            break
        else:
            print(f'crossing {min_ind} with rating {min_track}, case {min_case}')
            # adjust the skeleton!
            skeleton[min_ind[0]], skeleton[min_ind[1]] = manipulate_skeleton(
                segment1=skeleton[min_ind[0]],
                segment2=skeleton[min_ind[1]],
                bridgepoint1=min_bp1,
                bridgepoint2=min_bp2,
                case=min_case)
            done_or_flagged.append(min_ind)

        a = 0

    # build the updated skeleton
    for bone in skeleton:
        lines2obj([(bone.left, bone.right)], path=skeletonpath, topic=f'prc_{bone.name}')

    a = 0
