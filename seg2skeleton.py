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


import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import copy

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

        # skeleton = allocate_unsegmented_elements(skeleton, non_bone_segment_ids, cloud_df, config)

        # skeleton = allocate_unsegmented_elements_dual(skeleton, non_bone_segment_ids, cloud_df, config)

        # skeleton = allocate_unsegmented_elements_rev(skeleton, non_bone_segment_ids, cloud_df, config)
        return skeleton


import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import copy


def allocate_unsegmented_elements_rev(skeleton, non_bone_segments, cloud, config):
    """
    Unified allocation of unsegmented elements (patches and individual points) to skeleton segments
    using batch processing and intelligent scoring.

    Args:
        skeleton: Skeleton object containing bones/segments
        non_bone_segments: List of segment IDs that are not bones
        cloud: Point cloud DataFrame
        config: Configuration object

    Returns:
        Updated skeleton with allocated points
    """
    # Initialize spatial structures for segments
    segment_hulls = {}
    segment_kdtrees = {}
    segment_directions = {}
    bone_ids = []

    # Precompute segment spatial structures
    for bone in skeleton.bones:
        bone_id = int(bone.name.split('_')[1])
        if bone.break_flag is not None:
            non_bone_segments.append(bone_id)
        segment_hulls[bone_id] = ConvexHull(bone.points)
        segment_kdtrees[bone_id] = cKDTree(bone.points)
        segment_directions[bone_id] = bone.vector_3D
        bone_ids.append(bone_id)

    # Get unsegmented elements
    cloud_nonseg = cloud.loc[cloud['instance_pr'].isin(non_bone_segments)]

    # Create unified dataset of points to process
    points_to_process = pd.DataFrame()

    # Add patched points
    patched_points = cloud_nonseg[cloud_nonseg['ransac_patch'] != 0].copy()
    patched_points['is_patch'] = True
    patched_points['patch_id'] = patched_points['ransac_patch']

    # Add individual points
    individual_points = cloud[cloud['ransac_patch'] == 0].copy()
    individual_points['is_patch'] = False
    individual_points['patch_id'] = -1

    # Combine all points
    points_to_process = pd.concat([patched_points, individual_points])

    def compute_allocation_scores(chunk):
        """Compute allocation scores for a chunk of points."""
        scores = {}

        for bone_id in bone_ids:
            # Distance scores
            distances, _ = segment_kdtrees[bone_id].query(chunk[['x', 'y', 'z']].values)
            max_distance = config.skeleton.init_max_distance  # Using existing config parameter
            distance_scores = 1 - np.clip(distances / max_distance, 0, 1)

            # Angular scores
            normals = chunk[['nx', 'ny', 'nz']].values
            segment_dir = segment_directions[bone_id]

            # Vectorized angle calculation
            angles = np.arccos(np.clip(np.abs(np.dot(normals, segment_dir)), -1.0, 1.0))
            angles = np.minimum(angles, np.pi / 2 - angles) * 180 / np.pi

            # Use different angle thresholds based on whether points are patched or not
            max_angles = np.where(
                chunk['is_patch'],
                config.skeleton.init_max_angle_rn_sn,  # For patches
                config.skeleton.init_max_angle_n_sn  # For individual points
            )
            angle_scores = 1 - np.clip(angles / max_angles, 0, 1)

            # Confidence scores
            confidence_scores = chunk['confidence'].values

            # Combine scores with weights
            combined_scores = (
                    0.4 * distance_scores +
                    0.4 * angle_scores +
                    0.2 * confidence_scores
            )

            scores[bone_id] = combined_scores

        return scores

    # Process in batches
    batch_size = 1000
    allocation_results = []

    for batch_start in tqdm(range(0, len(points_to_process), batch_size), desc="Processing points"):
        batch_end = min(batch_start + batch_size, len(points_to_process))
        chunk = points_to_process.iloc[batch_start:batch_end]

        # Compute scores for all segments
        batch_scores = compute_allocation_scores(chunk)

        # Find best segment for each point
        best_segments = []
        best_scores = []

        for i in range(len(chunk)):
            point_scores = {seg_id: scores[i] for seg_id, scores in batch_scores.items()}
            best_segment = max(point_scores.items(), key=lambda x: x[1])

            # Only allocate if score meets minimum threshold (0.5 means both distance and angle criteria are reasonably met)
            if best_segment[1] > 0.5:  # Using fixed threshold instead of config
                best_segments.append(best_segment[0])
                best_scores.append(best_segment[1])
            else:
                best_segments.append(-1)  # Unallocated
                best_scores.append(0)

        # Store results
        chunk_results = chunk.copy()
        chunk_results['allocated_segment'] = best_segments
        chunk_results['allocation_score'] = best_scores
        allocation_results.append(chunk_results)

    # Combine results
    allocation_df = pd.concat(allocation_results)

    # Update skeleton and cloud
    for bone_id in bone_ids:
        # Get points allocated to this segment
        segment_points = allocation_df[allocation_df['allocated_segment'] == bone_id]

        if len(segment_points) > 0:
            bone = skeleton.get_bone(bone_id)

            # Update bone with new points
            bone.points = np.vstack((
                bone.points,
                segment_points[['x', 'y', 'z']].values.astype(np.float32)
            ))
            bone.normals = np.vstack((
                bone.normals,
                segment_points[['nx', 'ny', 'nz']].values.astype(np.float32)
            ))
            bone.points_data = pd.concat([bone.points_data, segment_points])

            # Update cloud
            cloud.loc[segment_points.index, 'instance_pr'] = bone_id

            # Recalculate bone axes
            bone.calc_axes(plot=False)

    # Log allocation statistics
    total_points = len(allocation_df)
    allocated_points = len(allocation_df[allocation_df['allocated_segment'] != -1])
    print(f"Allocated {allocated_points}/{total_points} points ({allocated_points / total_points * 100:.2f}%)")

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
        nonseg_patch_point_sn = cloud.loc[nonseg_patch_point, ['snx', 'sny', 'snz']].values

        if len(ranged_segments) != 0:
            for ranged_segment in ranged_segments:
                # retrieve bone
                segment = skeleton.get_bone(ranged_segment)
                segment_direction = segment.vector_3D
                # angle = angular_deviation(nonseg_patch_point_rn, segment_direction) % 90
                angle = angular_deviation(nonseg_patch_point_sn, segment_direction) % 180
                angle = min(angle, 180 - angle)

                if angle < config.skeleton.init_max_angle_rn_sn:
                    # patch is burnt
                    patch_point_ids = cloud.loc[cloud['ransac_patch'] == cloud.loc[nonseg_patch_point, 'ransac_patch'], 'id'].tolist()
                    # add points to segment
                    patch_point_data = cloud.loc[cloud['id'].isin(patch_point_ids)]
                    segment.points = np.vstack((segment.points, patch_point_data[['x', 'y', 'z']].values.astype(np.float32)))
                    segment.normals = np.vstack((segment.normals, patch_point_data[['nx', 'ny', 'nz']].values.astype(np.float32)))
                    segment.points_data = pd.concat([segment.points_data, patch_point_data])
                    segment.calc_axes(plot=False)
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
        point_supernormal = cloud.loc[point_id, ['snx', 'sny', 'snz']].values
        if len(ranged_segments) != 0:
            for ranged_segment in ranged_segments:
                segment = skeleton.get_bone(ranged_segment)
                segment_direction = segment.vector_3D
                angle = angular_deviation(point_normal, segment_direction) % 90
                angle = angular_deviation(point_supernormal, segment_direction) % 180
                angle = min(angle, 180 - angle)

                if angle < config.skeleton.init_max_angle_n_sn:
                    segment.points = np.vstack((segment.points, cloud.loc[point_id, ['x', 'y', 'z']].values.astype(np.float32)))
                    segment.normals = np.vstack((segment.normals, cloud.loc[point_id, ['nx', 'ny', 'nz']].values.astype(np.float32)))
                    segment.points_data = pd.concat([segment.points_data, cloud.loc[cloud['id'] == point_id]])
                    cloud.loc[point_id, 'instance_pr'] = ranged_segment
                    added += 1
                    break

    print(f'non-segmented points allocated: {added}')


    for i, bone in enumerate(skeleton.bones):
        print(f'bone {bone.name} recalc axes')
        bone.calc_axes(plot=False)

    return skeleton


def allocate_unsegmented_elements_dual(skeleton, non_bone_segments, cloud, config):
    """
    Modified version with dual criteria (supernormal AND ransac normal) for both patches and single points.
    """
    segment_hulls = {}
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

    # Process patches with dual criteria
    for nonseg_patch_point in tqdm(cloud_nonseg_patched_points, desc='allocating non-segmented patch points', total=len(cloud_nonseg_patched_points)):
        if nonseg_patch_point not in to_do:
            continue

        centerpoint = cloud.loc[nonseg_patch_point, ['x', 'y', 'z']].values.astype(np.float32)
        centerpoint = np.reshape(centerpoint, (3, 1))
        centerpoint = np.mean(centerpoint, axis=1)
        ranged_segments = point_to_hull_dict(nonseg_patch_point, cloud, segment_hulls, config, step='patch')

        # Get both normals for dual criteria
        patch_rn = cloud.loc[nonseg_patch_point, ['rnx', 'rny', 'rnz']].values
        patch_sn = cloud.loc[nonseg_patch_point, ['snx', 'sny', 'snz']].values

        if len(ranged_segments) != 0:
            for ranged_segment in ranged_segments:
                segment = skeleton.get_bone(ranged_segment)
                segment_direction = segment.vector_3D

                # Calculate angles for both normals
                angle_rn = angular_deviation(patch_rn, segment_direction) % 90
                angle_rn = min(angle_rn, 90 - angle_rn)

                angle_sn = angular_deviation(patch_sn, segment_direction) % 180
                angle_sn = min(angle_sn, 180 - angle_sn)

                # Check both criteria for patches
                if (angle_rn < config.skeleton.init_max_angle_rn_sn and
                        angle_sn < config.skeleton.init_max_angle_n_sn):  # Using n_sn threshold for supernormal

                    # Patch meets both criteria
                    patch_point_ids = cloud.loc[cloud['ransac_patch'] == cloud.loc[nonseg_patch_point, 'ransac_patch'], 'id'].tolist()

                    # add points to segment
                    patch_point_data = cloud.loc[cloud['id'].isin(patch_point_ids)]
                    segment.points = np.vstack((segment.points, patch_point_data[['x', 'y', 'z']].values.astype(np.float32)))
                    segment.normals = np.vstack((segment.normals, patch_point_data[['nx', 'ny', 'nz']].values.astype(np.float32)))
                    segment.points_data = pd.concat([segment.points_data, patch_point_data])
                    segment.calc_axes(plot=False)
                    cloud.loc[patch_point_ids, 'instance_pr'] = ranged_segment

                    # remove points from to_do
                    to_do_set = set(to_do)
                    to_do_set.difference_update(patch_point_ids)
                    to_do = list(to_do_set)
                    break

        else:
            non_allocated += 1

    print(f'non-segmented patch points not allocated: {non_allocated}')

    # Process individual points with dual criteria
    added = 0
    for point_id in tqdm(nonseg_point_ids, desc='allocating non-segmented points', total=len(nonseg_point_ids)):
        if added == 100:  # TODO: release breaks for full test!
            break

        ranged_segments = point_to_hull_dict(point_id, cloud, segment_hulls, config, step='single')

        # Get both normals for dual criteria
        point_normal = cloud.loc[point_id, ['nx', 'ny', 'nz']].values
        point_sn = cloud.loc[point_id, ['snx', 'sny', 'snz']].values

        if len(ranged_segments) != 0:
            for ranged_segment in ranged_segments:
                segment = skeleton.get_bone(ranged_segment)
                segment_direction = segment.vector_3D

                # Calculate angles for both normals
                angle_n = angular_deviation(point_normal, segment_direction) % 90
                angle_n = min(angle_n, 90 - angle_n)

                angle_sn = angular_deviation(point_sn, segment_direction) % 180
                angle_sn = min(angle_sn, 180 - angle_sn)

                # Check both criteria for single points
                if (angle_n < config.skeleton.init_max_angle_n_sn and
                        angle_sn < config.skeleton.init_max_angle_rn_sn):  # Using rn_sn threshold for supernormal

                    # Point meets both criteria
                    segment.points = np.vstack((segment.points, cloud.loc[point_id, ['x', 'y', 'z']].values.astype(np.float32)))
                    segment.normals = np.vstack((segment.normals, point_normal))
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

