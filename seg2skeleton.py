import itertools
import os
import pickle
from math import ceil, sqrt
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.spatial import ConvexHull

from structure.Cloud import Segment
from structure.Skeleton import Skeleton
from tools.geometry import warped_vectors_intersection, manipulate_skeleton, rotation_matrix_from_vectors
from tools.IO import lines2obj, cache_meta
from tools.local import angular_deviation
from tools.utils import update_logbook_checklist
from tools.test_plots import plot_test_in, plot_test_out


def inst2skeleton(cloud_df, config, df_cloud_flag=False, plot=True):
    # new approach has cloud with beams as dataframe. inject alternative from here
    skeleton = Skeleton(path=f'{str(os.getcwd())}/data/out_rev/skeleton',
                        types=['beams'],
                        config=config)  # beams only

    if skeleton.beams:
        segments: list = [f'beam_{i}' for i in np.unique(cloud_df['instance_pr']) if i != 0]
        non_bone_segment_ids = []

        for segment in segments:
            cloud = Segment(name=segment, config=config)
            if df_cloud_flag:
                cloud.load_from_df(cloud_df, segment)
            else:
                cloud.load_from_txt(segment)

            if len(cloud.points) > config.skeleton.init_min_count:
                print(f'- segment {segment}') # with initial size {len(cloud.points)}')
                cloud.calc_axes(plot=plot)
                if not cloud.break_flag:
                    skeleton.add_cloud(cloud)
                else:
                    print(f'segment {segment} benched due to failed orientation estimation')
                    non_bone_segment_ids.append(int(segment.split('_')[1]))
            else:
                print(f'segment {segment} benched due to initial size')
                non_bone_segment_ids.append(int(segment.split('_')[1]))

        skeleton = allocate_unsegmented_elements(skeleton, non_bone_segment_ids, cloud_df, config)

        return skeleton


def allocate_unsegmented_elements(skeleton, non_bone_segments, cloud, config):
    segment_hulls = {}  # dictionary segment convex hulls
    bone_ids = []
    for bone in skeleton.bones:
        bone_id = int(bone.name.split('_')[1])
        if bone.break_flag is not None:
            non_bone_segments.append(bone_id)
        segment_hulls[bone_id] = ConvexHull(bone.points)
        bone_ids.append(bone_id)

    # identify non-segmented planar patches
    cloud_nonseg = cloud.loc[cloud['instance_pr'].isin(non_bone_segments)]
    cloud_nonseg_patched = cloud_nonseg[cloud_nonseg['ransac_patch'] != 0]
    cloud_nonseg_patched_points = cloud_nonseg_patched['id'].tolist()
    nonseg_patch_ids = cloud_nonseg_patched['ransac_patch'].unique().tolist()


    cloud_nonseg_nopatch_points = cloud_nonseg[cloud_nonseg['ransac_patch'] == 0]
    nonseg_point_ids = cloud_nonseg_nopatch_points['id'].tolist()





    for nonseg_patch_point in tqdm(cloud_nonseg_patched_points, desc='allocating non-segmented patch points', total=len(cloud_nonseg_patched_points)):
        ranged_segments = point_to_hull_dict(nonseg_patch_point, cloud, segment_hulls, config)
        # test angle between point ransac normal and segment supernormal
        nonseg_patch_point_rn = cloud.loc[nonseg_patch_point, ['rnx', 'rny', 'rnz']].values

        if len(ranged_segments) != 0:
            for ranged_segment in ranged_segments:
                # retrieve bone
                segment = skeleton.get_bone(ranged_segment)
                segment_direction = segment.line_raw_dir
                angle = angular_deviation(nonseg_patch_point_rn, segment_direction) % 90
                angle = min(angle, 90 - angle)

                if angle < config.skeleton.init_max_angle_rn_sn:
                    # add point to segment
                    segment.points = np.vstack((segment.points, cloud.loc[nonseg_patch_point, ['x', 'y', 'z']].values.astype(np.float32)))
                    segment.points_data = pd.concat([segment.points_data, cloud.loc[nonseg_patch_point, :].to_frame().T])
                    cloud.loc[nonseg_patch_point, 'instance_pr'] = ranged_segment
                    break
            # break
        else:
            print(f'point {nonseg_patch_point} not allocated to any segment')

    for i, bone in enumerate(skeleton.bones):
        print(f'bone {bone.name} recalc axes')
        bone.calc_axes(plot=False)

    skeleton.plot_cog_skeleton()

    a = 0


def point_to_hull_dict(point_id, cloud, hull_dict, config):
    segment_dists = {}
    for segment_hull in hull_dict:
        hull = hull_dict[segment_hull]
        point = cloud.loc[point_id, ['x', 'y', 'z']].values
        dists = np.abs(hull.equations.dot(np.append(point, 1))) # abs to avoid negative distances, dont care abt inner/outer
        min_dist = np.min(dists)
        if min_dist < config.skeleton.allocate_max_dist:
            segment_dists[segment_hull] = np.min(dists)

    # return a list of segments ranked by distance, smallest first
    sorted_segment_ids = [k for k, v in sorted(segment_dists.items(), key=lambda item: item[1])]

    return sorted_segment_ids


if __name__ == '__main__':
    inst2skeleton()

