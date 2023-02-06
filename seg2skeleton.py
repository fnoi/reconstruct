import itertools
import os

from structure.CloudSegment import CloudSegment
from structure.geometry import vector_distance, manipulate_skeleton
from tools.IO import points2txt, lines2obj

if __name__ == '__main__':

    skeletonpath = f'{str(os.getcwd())}/data/out/0_skeleton'
    if not os.path.exists(skeletonpath):
        os.makedirs(skeletonpath)

    segments: list = [
        'beam_01',
        'beam_02',
        #'beam_03',
        'beam_04',
        'beam_05',
        'beam_06',
        'beam_07',
        'beam_08',
        'beam_09',
        #'beam_10'
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
        lines2obj([(bone.left, bone.right)], path=skeletonpath, topic=f'raw_{bone.name}')

    done_or_flagged = []

    while True:
        min_ind = None
        min_track = 1e10
        neighbors = list(itertools.combinations(range(len(skeleton)), 2))
        for neighbor in neighbors:
            bridgepoint1, bridgepoint2, rating, case = vector_distance(neighbor, skeleton[neighbor[0]], skeleton[neighbor[1]])
            if rating < min_track and neighbor not in done_or_flagged:
                min_track = rating
                min_ind = neighbor
                min_bp1 = bridgepoint1
                min_bp2 = bridgepoint2
                min_case = case
            # print(dist)
            # lines2obj([(pt1, pt2)], path=skeletonpath, topic=f'cross_me_{str(neighbor)}')

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


    a=0