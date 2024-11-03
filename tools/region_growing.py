import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

pd.options.mode.copy_on_write = True

from tools.local import (
    neighborhood_search,
    supernormal_svd_s1,
    consistency_flip,
    angular_deviation,
    supernormal_confidence,
    subset_cluster_neighbor_search
)
from tools.visual import rg_plot_active, rg_plot_what, rg_plot_finished


def get_seed_point(cloud, source_point_ids, sink_patch_ids, min_patch_size):
    """Get seed point for new cluster growth"""
    source_cloud = cloud[cloud['id'].isin(source_point_ids)]
    source_cloud.sort_values(by='confidence', ascending=False, inplace=True)

    for _, row in source_cloud.iterrows():
        patch_size = len(cloud[cloud['ransac_patch'] == row['ransac_patch']])
        if (row['ransac_patch'] != 0 and
                row['ransac_patch'] not in sink_patch_ids and
                patch_size >= min_patch_size):
            return (row['id'],
                    row['ransac_patch'],
                    row['confidence'],
                    np.asarray(row[['snx', 'sny', 'snz']].values))
    return None, None, None, None


def calculate_cluster_normals(cloud, reduced_cloud, ship_neighbors, seed_point_id,
                              seed_sn, seed_confidence, active_point_ids):
    """Calculate cluster normal vectors and confidence"""
    __normals = np.asarray(reduced_cloud.loc[reduced_cloud['id'].isin(ship_neighbors)][['nx', 'ny', 'nz']])
    cluster_sn, _s1, _s2, _s3 = supernormal_svd_s1(__normals, full_return=True)
    cluster_confidence = supernormal_confidence(cluster_sn, __normals, _s1, _s2, _s3)

    if seed_confidence > cluster_confidence:
        cluster_sn = seed_sn
        cluster_confidence = seed_confidence

    _cloud = cloud.loc[cloud['id'].isin(active_point_ids)]
    _patch_id = _cloud['ransac_patch'].mode().iloc[0]
    first_row = _cloud[_cloud['ransac_patch'] == _patch_id].iloc[0]
    cluster_rn = np.asarray(first_row[['rnx', 'rny', 'rnz']].values)

    return cluster_sn, cluster_rn, cluster_confidence


def evaluate_neighbor_patch(cloud, neighbor_patch, cluster_sn, cluster_rn, config, high_confidence_mode=False):
    """Evaluate if neighbor patch should be added to cluster"""
    neighbor_patch_sns = consistency_flip(
        np.asarray(cloud.loc[cloud['ransac_patch'] == neighbor_patch][['snx', 'sny', 'snz']].values)
    )
    neighbor_patch_sn = np.mean(neighbor_patch_sns, axis=0)
    neighbor_patch_first = cloud.loc[cloud['ransac_patch'] == neighbor_patch].iloc[0]
    neighbor_patch_rn = neighbor_patch_first[['rnx', 'rny', 'rnz']].values
    neighbor_patch_csn = cloud.loc[neighbor_patch_first['id'], ['csnx', 'csny', 'csnz']].values

    deviation_sn = min(angular_deviation(cluster_sn, neighbor_patch_csn) % 180,
                       180 - (angular_deviation(cluster_sn, neighbor_patch_csn) % 180))

    if high_confidence_mode:
        should_add = deviation_sn < config.region_growing.supernormal_angle_deviation_point
    else:
        deviation_rn = min(angular_deviation(cluster_rn, neighbor_patch_rn) % 90,
                           90 - (angular_deviation(cluster_rn, neighbor_patch_rn) % 90))
        should_add = (deviation_sn < config.region_growing.supernormal_angle_deviation_patch and
                      deviation_rn < config.region_growing.ransacnormal_angle_deviation_patch)

    return (should_add, neighbor_patch_sn, neighbor_patch_rn, neighbor_patch_csn,
            deviation_sn, deviation_rn if not high_confidence_mode else None)


def evaluate_neighbor_point(cloud, point_id, cluster_sn, config):
    """Evaluate if a single point should be added in high confidence mode"""
    point_csn = cloud.loc[point_id, ['csnx', 'csny', 'csnz']].values
    deviation_sn = min(angular_deviation(cluster_sn, point_csn) % 180,
                       180 - (angular_deviation(cluster_sn, point_csn) % 180))
    return deviation_sn < config.region_growing.supernormal_angle_deviation_point


def process_high_confidence_neighbors(cloud, neighbor_ids, cluster_sn, config, source_ids_point, source_ids_patch):
    """Process neighbors in high confidence mode, adding points and their patches"""
    added_point_ids = []
    added_patch_ids = []
    patches_to_process = set()

    # First pass: identify qualifying points and their patches
    for point_id in neighbor_ids:
        if evaluate_neighbor_point(cloud, point_id, cluster_sn, config):
            added_point_ids.append(point_id)
            patch_id = cloud.loc[point_id, 'ransac_patch']
            if patch_id != 0:  # Only add non-zero patch IDs
                patches_to_process.add(patch_id)

    # Second pass: add all points from qualifying patches
    for patch_id in patches_to_process:
        if patch_id in source_ids_patch:  # Only process patches that are still in source
            patch_points = cloud[cloud['ransac_patch'] == patch_id]['id'].tolist()
            added_point_ids.extend(patch_points)
            added_patch_ids.append(patch_id)

    # Remove duplicates and ensure all points are in source
    added_point_ids = list(set(added_point_ids) & set(source_ids_point))
    added_patch_ids = list(set(added_patch_ids))

    return added_point_ids, added_patch_ids


def region_growing_main(cloud, config):
    """Main region growing function"""
    if 'id' not in cloud.columns:
        cloud['id'] = range(len(cloud))
    cloud['instance_pr'] = 0

    source_ids_point = cloud['id'].to_list()
    source_ids_patch = cloud['ransac_patch'].unique().tolist()
    sink_ids_patch = []

    counter_patch = 0

    while len(source_ids_point) > config.region_growing.leftover_relative * len(cloud):
        counter_patch += 1
        print(f'Cluster {counter_patch}')

        # Get seed point
        seed_id_point, seed_id_patch, seed_confidence, seed_sn = get_seed_point(
            cloud, source_ids_point, sink_ids_patch, config.region_growing.min_patch_size)
        if seed_id_point is None:
            break

        # Check if seed point has high confidence
        high_confidence_mode = seed_confidence > 5
        if high_confidence_mode:
            print(f"High confidence seed point (confidence: {seed_confidence}). Using relaxed criteria.")

        # Initialize active and inactive sets
        active_point_ids = cloud[cloud['ransac_patch'] == seed_id_patch]['id'].to_list()
        active_patch_ids = [seed_id_patch]
        inactive_point_ids = []
        inactive_patch_ids = []

        # Update source sets
        source_ids_point = list(set(source_ids_point) - set(active_point_ids))
        source_ids_patch = list(set(source_ids_patch) - {seed_id_patch})

        # Initial neighbor check for high confidence seeds
        if high_confidence_mode:
            initial_neighbors, initial_cloud = subset_cluster_neighbor_search(cloud, active_point_ids, config)
            if initial_neighbors:
                cluster_sn, _, _ = calculate_cluster_normals(
                    cloud, initial_cloud, list(set(initial_neighbors)),
                    seed_id_point, seed_sn, seed_confidence, active_point_ids)

                # Add qualifying points and their patches
                new_points, new_patches = process_high_confidence_neighbors(
                    cloud, initial_neighbors, cluster_sn, config, source_ids_point, source_ids_patch
                )

                if new_points:
                    active_point_ids.extend(new_points)
                    active_patch_ids.extend(new_patches)
                    source_ids_point = list(set(source_ids_point) - set(new_points))
                    source_ids_patch = list(set(source_ids_patch) - set(new_patches))
                    print(f"High confidence mode added {len(new_points)} points and {len(new_patches)} patches")

        # Continue with normal region growing
        segment_iter = 0
        chk_log_patches = []
        chk_log_points = []
        floating_points_dict = {}

        while True:
            segment_iter += 1
            print(f'growth iter {segment_iter}')

            # Find neighbors
            neighbor_ids_cluster, neighbor_cloud_cluster = subset_cluster_neighbor_search(cloud, active_point_ids, config)
            if not neighbor_ids_cluster:
                sink_ids_patch.extend(active_patch_ids)
                cloud.loc[cloud['id'].isin(active_point_ids), 'instance_pr'] = counter_patch
                break

            neighbor_ids_cluster = list(set(neighbor_ids_cluster) - set(active_point_ids))
            if not neighbor_ids_cluster:
                sink_ids_patch.extend(active_patch_ids)
                cloud.loc[cloud['id'].isin(active_point_ids), 'instance_pr'] = counter_patch
                break

            # Calculate normals
            cluster_sn, cluster_rn, cluster_conf = calculate_cluster_normals(
                cloud, neighbor_cloud_cluster, list(set(neighbor_ids_cluster)),
                seed_id_point, seed_sn, seed_confidence, active_point_ids)

            # Process neighbors
            neighbor_ids_patch = neighbor_cloud_cluster.loc[neighbor_cloud_cluster['id'].isin(neighbor_ids_cluster)]['ransac_patch'].unique().tolist()
            if 0 in neighbor_ids_patch:
                neighbor_ids_patch.remove(0)
            neighbor_ids_points_unpatched = neighbor_cloud_cluster[neighbor_cloud_cluster['ransac_patch'] == 0]['id'].to_list()

            # Check for growth completion
            chk_neighbors_0 = chk_log_patches == len(neighbor_ids_patch)
            chk_log_patches = len(neighbor_ids_patch)
            chk_neighbors_1 = chk_log_points == len(neighbor_ids_points_unpatched)
            chk_log_points = len(neighbor_ids_points_unpatched)

            if chk_neighbors_0 and chk_neighbors_1:
                floating_points_dict[seed_id_patch] = neighbor_ids_points_unpatched
                sink_ids_patch.extend(active_patch_ids)
                cloud.loc[cloud['id'].isin(active_point_ids), 'instance_pr'] = counter_patch
                break

            # Evaluate neighbor patches (using normal criteria)
            for neighbor_patch in neighbor_ids_patch:
                if (neighbor_patch in sink_ids_patch or
                        neighbor_patch in active_patch_ids or
                        neighbor_patch in inactive_patch_ids or
                        neighbor_patch not in source_ids_patch):
                    continue

                should_add, n_patch_sn, n_patch_rn, n_patch_csn, dev_sn, dev_rn = evaluate_neighbor_patch(
                    cloud, neighbor_patch, cluster_sn, cluster_rn, config, high_confidence_mode=False)

                point_ids = cloud[cloud['ransac_patch'] == neighbor_patch]['id'].to_list()
                active_plot = copy.deepcopy(active_point_ids)

                if should_add:
                    active_point_ids.extend(point_ids)
                    active_patch_ids.append(neighbor_patch)
                    source_ids_point = list(set(source_ids_point) - set(active_point_ids))
                    source_ids_patch.remove(neighbor_patch)
                    source_ids_patch = list(set(source_ids_patch))
                else:
                    inactive_point_ids.extend(point_ids)
                    inactive_patch_ids.append(neighbor_patch)

                print(f'active: {len(active_point_ids)}, inactive: {len(inactive_point_ids)}, source: {len(source_ids_point)}')

        # plot points: active green, inactive red, source yellow, sink grey
        rg_plot_finished(cloud=cloud, active_points=active_point_ids, inactive_points=inactive_point_ids,
                         source_points=source_ids_point, sink_points_patch_ids=sink_ids_patch)

    return cloud