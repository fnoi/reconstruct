import copy
import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

from tools.local import (
    neighborhood_search,
    supernormal_svd_s1,
    consistency_flip,
    angular_deviation,
    supernormal_confidence,
    subset_cluster_neighbor_search
)
from tools.visual import rg_plot_active, rg_plot_what


def get_seed_point(cloud, source_point_ids, sink_patch_ids, min_patch_size):
    """Get seed point for new cluster growth"""
    source_cloud = cloud[cloud['id'].isin(source_point_ids)]
    source_cloud.sort_values(by='csn_confidence', ascending=True, inplace=True)

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


def calculate_cluster_normals(cloud, reduced_cloud, ship_neighbors, seed_point_id, seed_sn, seed_confidence, active_point_ids):
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


def evaluate_neighbor_patch(cloud, neighbor_patch, cluster_sn, cluster_rn, config):
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
    deviation_rn = min(angular_deviation(cluster_rn, neighbor_patch_rn) % 90,
                       90 - (angular_deviation(cluster_rn, neighbor_patch_rn) % 90))

    should_add = (deviation_sn < config.region_growing.supernormal_angle_deviation_patch and
                  deviation_rn < config.region_growing.ransacnormal_angle_deviation_patch)

    return (should_add, neighbor_patch_sn, neighbor_patch_rn, neighbor_patch_csn,
            deviation_sn, deviation_rn)


def region_growing_main(cloud, config):
    """Main region growing function"""
    if 'id' not in cloud.columns:
        cloud['id'] = range(len(cloud))
    cloud['instance_pr'] = 0

    source_ids_point = cloud['id'].to_list()
    source_ids_patch = cloud['ransac_patch'].unique().tolist()
    sink_ids_patch = []

    counter_patch = 0
    plot_count = 0

    while len(source_ids_point) > config.region_growing.leftover_relative * len(cloud):
        counter_patch += 1
        print(f'Cluster {counter_patch}')

        # Get seed point
        seed_point_id, seed_patch_id, seed_confidence, seed_sn = get_seed_point(
            cloud, source_ids_point, sink_ids_patch, config.region_growing.min_patch_size)
        if seed_point_id is None:
            break

        # Initialize active and inactive sets
        active_point_ids = cloud[cloud['ransac_patch'] == seed_patch_id]['id'].to_list()
        active_patch_ids = [seed_patch_id]
        inactive_point_ids = []
        inactive_patch_ids = []

        # Update source sets
        source_ids_point = list(set(source_ids_point) - {seed_point_id})
        source_ids_patch = list(set(source_ids_patch) - {seed_patch_id})

        # Grow cluster
        segment_iter = 0
        chk_log_patches = []
        chk_log_points = []
        floating_points_dict = {}

        while True:
            segment_iter += 1
            print(f'growth iter {segment_iter}')

            # Find neighbors
            cluster_neighbors, reduced_cloud = subset_cluster_neighbor_search(cloud, active_point_ids, config)
            if not cluster_neighbors:
                break

            cluster_neighbors = list(set(cluster_neighbors) - set(active_point_ids))

            # Calculate normals
            cluster_sn, cluster_rn, cluster_confidence = calculate_cluster_normals(
                cloud, reduced_cloud, list(set(cluster_neighbors)),
                seed_point_id, seed_sn, seed_confidence, active_point_ids)

            # Process neighbors
            neighbor_patch_ids = reduced_cloud.loc[reduced_cloud['id'].isin(cluster_neighbors)]['ransac_patch'].unique().tolist()
            if 0 in neighbor_patch_ids:
                neighbor_patch_ids.remove(0)
            neighbor_unpatched_point_ids = reduced_cloud[reduced_cloud['ransac_patch'] == 0]['id'].to_list()

            # Check for growth completion
            chk_neighbors_0 = chk_log_patches == len(neighbor_patch_ids)
            chk_log_patches = len(neighbor_patch_ids)
            chk_neighbors_1 = chk_log_points == len(neighbor_unpatched_point_ids)
            chk_log_points = len(neighbor_unpatched_point_ids)

            if chk_neighbors_0 and chk_neighbors_1:
                floating_points_dict[seed_patch_id] = neighbor_unpatched_point_ids
                sink_ids_patch.extend(active_patch_ids)
                cloud.loc[cloud['id'].isin(active_point_ids), 'instance_pr'] = counter_patch
                rg_plot_active(cloud, active_point_ids, counter_patch)
                break

            # Evaluate neighbor patches
            for neighbor_patch in neighbor_patch_ids:
                if (neighbor_patch in sink_ids_patch or
                        neighbor_patch in active_patch_ids or
                        neighbor_patch in inactive_patch_ids or
                        neighbor_patch not in source_ids_patch):
                    continue

                should_add, n_patch_sn, n_patch_rn, n_patch_csn, dev_sn, dev_rn = evaluate_neighbor_patch(
                    cloud, neighbor_patch, cluster_sn, cluster_rn, config)

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

    return cloud