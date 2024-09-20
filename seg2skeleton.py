import copy
import itertools
import os
import pickle
from math import ceil, sqrt
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

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
    """
    1. allocate unsegmented patches to the closest segments based on ransac normal/segment orientation angular difference and max distance
    2. allocate unsegmented unpatched points to the closest segments based on normal / segment orientation angular difference and max distance
    fast distance computation using convex hulls
    """
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

    cloud_nonseg_nopatch_points = cloud[cloud['ransac_patch'] == 0]
    nonseg_point_ids = cloud_nonseg_nopatch_points['id'].tolist()

    non_allocated = 0
    to_do = copy.deepcopy(cloud_nonseg_patched_points)

    for nonseg_patch_point in tqdm(cloud_nonseg_patched_points, desc='allocating non-segmented patch points', total=len(cloud_nonseg_patched_points)):
        if nonseg_patch_point not in to_do:
            continue
        # identify centerpoint of patch
        centerpoint = cloud.loc[nonseg_patch_point, ['x', 'y', 'z']].values.astype(np.float32)
        centerpoint = np.reshape(centerpoint, (3, 1))
        centerpoint = np.mean(centerpoint, axis=1)
        ranged_segments = point_to_hull_dict(nonseg_patch_point, cloud, segment_hulls, config, step='patch')
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
                    # patch is burnt
                    patch_point_ids = cloud.loc[cloud['ransac_patch'] == cloud.loc[nonseg_patch_point, 'ransac_patch'], 'id'].tolist()
                    # add points to segment
                    patch_point_data = cloud.loc[cloud['id'].isin(patch_point_ids)]
                    segment.points = np.vstack((segment.points, patch_point_data[['x', 'y', 'z']].values.astype(np.float32)))
                    segment.points_data = pd.concat([segment.points_data, patch_point_data])
                    cloud.loc[nonseg_patch_point, 'instance_pr'] = ranged_segment
                    # remove points from to_do
                    to_do = [x for x in to_do if x not in patch_point_ids]


                    break

        else:
            non_allocated += 1

    print(f'non-segmented patch points not allocated: {non_allocated}')

    added = 0
    for point_id in tqdm(nonseg_point_ids, desc='allocating non-segmented points', total=len(nonseg_point_ids)):
        if added == 100:  # TODO: release breaks for full test!
            break
        ranged_segments = point_to_hull_dict(point_id, cloud, segment_hulls, config, step='single')
        point_normal = cloud.loc[point_id, ['nx', 'ny', 'nz']].values
        if len(ranged_segments) != 0:
            for ranged_segment in ranged_segments:
                segment = skeleton.get_bone(ranged_segment)
                segment_direction = segment.line_raw_dir
                angle = angular_deviation(point_normal, segment_direction) % 90
                angle = min(angle, 90 - angle)

                if angle < config.skeleton.init_max_angle_n_sn:
                    segment.points = np.vstack((segment.points, cloud.loc[point_id, ['x', 'y', 'z']].values.astype(np.float32)))
                    segment.points_data = pd.concat([segment.points_data, cloud.loc[cloud['id'] == point_id]])
                    cloud.loc[point_id, 'instance_pr'] = ranged_segment
                    added += 1
                    break

    print(f'non-segmented points allocated: {added}')


    for i, bone in enumerate(skeleton.bones):
        print(f'bone {bone.name} recalc axes')
        bone.calc_axes(plot=False)

    return skeleton


def point_to_hull_dict(point_id, cloud, hull_dict, config, step='patch'):
    segment_dists = {}
    for segment_hull in hull_dict:
        hull = hull_dict[segment_hull]

        point = cloud.loc[point_id, ['x', 'y', 'z']].values.astype(np.float32)

        min_dist = point_to_hull_distance(point, hull)

        debug_plot = False
        if debug_plot:
            visualize_hull_plotly(
                hull_points=hull.points, hull=hull, point_cloud=cloud.loc[:, ['x', 'y', 'z']].values.astype(np.float32),
                ref_point=cloud.loc[point_id, ['x', 'y', 'z']].values.astype(np.float32), mindist=min_dist
            )

        if step == 'patch':
            dist_threshold = config.skeleton.allocate_max_dist_patch
        elif step == 'single':
            dist_threshold = config.skeleton.allocate_max_dist_point
        else:
            raise ValueError(f'step {step} not recognized')

        if min_dist < dist_threshold:
            segment_dists[segment_hull] = min_dist

    # return a list of segments ranked by distance, smallest first
    sorted_segment_ids = [k for k, v in sorted(segment_dists.items(), key=lambda item: item[1])]



    return sorted_segment_ids


def point_to_hull_distance(point, hull):
    """
    Calculate the true minimum distance from a point to a convex hull.
    This accounts for distances to facets, edges, and vertices of any dimension.
    """
    hull_points = hull.points[hull.vertices]

    # Distance to vertices
    distances_to_vertices = cdist([point], hull_points).min()
    min_distance = distances_to_vertices

    # Function to calculate distance to a line segment
    def point_to_line_segment(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0, min(1, t))  # Clamp t to [0, 1]
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    # Check distance to each edge
    for simplex in hull.simplices:
        n = len(simplex)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = hull.points[simplex[i]], hull.points[simplex[j]]
                distance = point_to_line_segment(point, p1, p2)
                min_distance = min(min_distance, distance)

    return min_distance



def visualize_hull_plotly(hull_points, hull, point_cloud, ref_point, mindist):
    # Crate scatter of full point cloud
    scatter_0 = go.Scatter3d(
        x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
        mode='markers',
        marker=dict(size=1, color='grey'),
        name='Point Cloud'
    )

    # reshape ref_point to (3,1) array
    ref_point = np.reshape(ref_point, (3,1))
    # Create scatter of reference point
    ref_point_scatter = go.Scatter3d(
        x=ref_point[0], y=ref_point[1], z=ref_point[2],
        mode='markers',
        marker=dict(size=6, color='black', symbol='diamond'),
        name='Reference Point'
    )


    # Create scatter plot of original points
    scatter = go.Scatter3d(
        x=hull_points[:, 0], y=hull_points[:, 1], z=hull_points[:, 2],
        mode='markers',
        marker=dict(size=4, color='blue'),
        name='Original Points'
    )

    # Create lines for the edges of the convex hull
    lines = []
    for simplex in hull.simplices:
        lines.append(go.Scatter3d(
            x=hull_points[simplex, 0],
            y=hull_points[simplex, 1],
            z=hull_points[simplex, 2],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        ))

    # Create scatter plot of hull points
    hull_points = hull_points[hull.vertices]
    hull_scatter = go.Scatter3d(
        x=hull_points[:, 0], y=hull_points[:, 1], z=hull_points[:, 2],
        mode='markers',
        marker=dict(size=6, color='red', symbol='diamond'),
        name='Hull Points'
    )

    # Combine all traces
    data = [scatter, hull_scatter, scatter_0, ref_point_scatter] + lines

    # Set up the layout
    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title=f'mindist: {mindist}',
    )

    # Create and show the figure
    fig = go.Figure(data=data, layout=layout)
    fig.show()


if __name__ == '__main__':
    inst2skeleton()

