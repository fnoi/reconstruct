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
    if cluster_sn is not None:
        cluster_confidence = supernormal_confidence(cluster_sn, __normals, _s1, _s2, _s3)

        if seed_confidence > cluster_confidence:
            cluster_sn = seed_sn
            cluster_confidence = seed_confidence

    else:
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


# def process_high_confidence_neighbors(cloud, neighbor_ids, cluster_sn, config, source_ids_point, source_ids_patch):
#     """Process neighbors in high confidence mode, adding points and their patches"""
#     added_point_ids = []
#     added_patch_ids = []
#     patches_to_process = set()
#
#     # First pass: identify qualifying points and their patches
#     for point_id in neighbor_ids:
#         if evaluate_neighbor_point(cloud, point_id, cluster_sn, config):
#             added_point_ids.append(point_id)
#             patch_id = cloud.loc[point_id, 'ransac_patch']
#             if patch_id != 0:  # Only add non-zero patch IDs
#                 patches_to_process.add(patch_id)
#
#     # Second pass: add all points from qualifying patches
#     for patch_id in patches_to_process:
#         if patch_id in source_ids_patch:  # Only process patches that are still in source
#             patch_points = cloud[cloud['ransac_patch'] == patch_id]['id'].tolist()
#             added_point_ids.extend(patch_points)
#             added_patch_ids.append(patch_id)
#
#     # Remove duplicates and ensure all points are in source
#     added_point_ids = list(set(added_point_ids) & set(source_ids_point))
#     added_patch_ids = list(set(added_patch_ids))
#
#     return added_point_ids, added_patch_ids


def process_high_confidence_neighbors(cloud, neighbor_ids, cluster_sn, config, source_ids_point, source_ids_patch):
    """Process neighbors in high confidence mode with memory efficiency"""
    # Use sets for faster operations and memory efficiency
    neighbor_patches = set(cloud.loc[neighbor_ids, 'ransac_patch'].unique()) - {0}
    neighbor_patches &= set(source_ids_patch)  # Intersection with source patches

    added_point_ids = set()
    added_patch_ids = set()

    for patch_id in neighbor_patches:
        patch_mask = cloud['ransac_patch'] == patch_id
        patch_points = cloud.loc[patch_mask]

        # Check point normals without storing all points
        qualifying_count = 0
        total_points = 0

        for _, point in patch_points.iterrows():
            if point['id'] not in source_ids_point:
                continue
            total_points += 1
            point_csn = point[['csnx', 'csny', 'csnz']].values
            deviation_sn = min(angular_deviation(cluster_sn, point_csn) % 180,
                               180 - (angular_deviation(cluster_sn, point_csn) % 180))
            if deviation_sn < config.region_growing.supernormal_angle_deviation_point:
                qualifying_count += 1

        if total_points > 0 and qualifying_count / total_points > 0.75:
            # Get point IDs efficiently using boolean indexing
            valid_points = set(patch_points['id']) & set(source_ids_point)
            added_point_ids.update(valid_points)
            added_patch_ids.add(patch_id)

    return list(added_point_ids), list(added_patch_ids)


def region_growing_main(cloud, config):
    """Memory-optimized main region growing function"""
    if 'id' not in cloud.columns:
        cloud['id'] = range(len(cloud))
    cloud['instance_pr'] = 0

    # Use sets for faster operations and memory efficiency
    source_ids_point = set(cloud['id'])
    source_ids_patch = set(cloud['ransac_patch'].unique()) - {0}
    sink_ids_patch = set()

    counter_patch = 0

    while len(source_ids_point) > config.region_growing.leftover_relative * len(cloud):
        counter_patch += 1
        print(f'Cluster {counter_patch}')

        # Get seed point
        seed_result = get_seed_point(cloud, source_ids_point, sink_ids_patch, config.region_growing.min_patch_size)
        if seed_result is None:
            break
        seed_id_point, seed_id_patch, seed_confidence, seed_sn = seed_result

        # Check if seed point has high confidence
        high_confidence_mode = seed_confidence > 5
        if high_confidence_mode:
            print(f"High confidence seed point (confidence: {seed_confidence})")

        # Initialize active and inactive sets as sets
        active_point_ids = set(cloud[cloud['ransac_patch'] == seed_id_patch]['id'])
        active_patch_ids = {seed_id_patch}
        inactive_point_ids = set()
        inactive_patch_ids = set()

        # Update source sets efficiently
        source_ids_point -= active_point_ids
        source_ids_patch.remove(seed_id_patch)

        # Process high confidence seeds
        if high_confidence_mode:
            initial_neighbors, initial_cloud = subset_cluster_neighbor_search(cloud, active_point_ids, config)
            if initial_neighbors:
                cluster_sn, _, _ = calculate_cluster_normals(
                    cloud, initial_cloud, initial_neighbors,
                    seed_id_point, seed_sn, seed_confidence, active_point_ids)

                new_points, new_patches = process_high_confidence_neighbors(
                    cloud, initial_neighbors, cluster_sn, config, source_ids_point, source_ids_patch
                )

                if new_points:
                    active_point_ids.update(new_points)
                    active_patch_ids.update(new_patches)
                    source_ids_point -= set(new_points)
                    source_ids_patch -= set(new_patches)

        # Regular growth phase
        segment_iter = 0
        prev_patches_count = 0
        prev_points_count = 0

        while True:
            segment_iter += 1
            print(f'growth iter {segment_iter}')

            # Find neighbors efficiently
            neighbor_ids_cluster, neighbor_cloud_cluster = subset_cluster_neighbor_search(
                cloud, active_point_ids, config)

            if not neighbor_ids_cluster:
                sink_ids_patch.update(active_patch_ids)
                cloud.loc[cloud['id'].isin(active_point_ids), 'instance_pr'] = counter_patch
                break

            neighbor_ids_cluster = set(neighbor_ids_cluster) - active_point_ids
            if not neighbor_ids_cluster:
                sink_ids_patch.update(active_patch_ids)
                cloud.loc[cloud['id'].isin(active_point_ids), 'instance_pr'] = counter_patch
                break

            # Calculate normals
            cluster_sn, cluster_rn, cluster_conf = calculate_cluster_normals(
                cloud, neighbor_cloud_cluster, neighbor_ids_cluster,
                seed_id_point, seed_sn, seed_confidence, active_point_ids)

            # Process neighbors efficiently
            neighbor_ids_patch = set(neighbor_cloud_cluster.loc[
                                         neighbor_cloud_cluster['id'].isin(neighbor_ids_cluster)
                                     ]['ransac_patch'].unique()) - {0}

            neighbor_ids_points_unpatched = set(
                neighbor_cloud_cluster[neighbor_cloud_cluster['ransac_patch'] == 0]['id']
            )

            # Check for growth completion
            if (len(neighbor_ids_patch) == prev_patches_count and
                    len(neighbor_ids_points_unpatched) == prev_points_count):
                sink_ids_patch.update(active_patch_ids)
                cloud.loc[cloud['id'].isin(active_point_ids), 'instance_pr'] = counter_patch
                break

            prev_patches_count = len(neighbor_ids_patch)
            prev_points_count = len(neighbor_ids_points_unpatched)

            # Evaluate neighbor patches efficiently
            for neighbor_patch in neighbor_ids_patch:
                if (neighbor_patch in sink_ids_patch or
                        neighbor_patch in active_patch_ids or
                        neighbor_patch in inactive_patch_ids or
                        neighbor_patch not in source_ids_patch):
                    continue

                should_add, *_ = evaluate_neighbor_patch(
                    cloud, neighbor_patch, cluster_sn, cluster_rn, config)

                point_ids = set(cloud[cloud['ransac_patch'] == neighbor_patch]['id'])

                if should_add:
                    active_point_ids.update(point_ids)
                    active_patch_ids.add(neighbor_patch)
                    source_ids_point -= point_ids
                    source_ids_patch.remove(neighbor_patch)
                else:
                    inactive_point_ids.update(point_ids)
                    inactive_patch_ids.add(neighbor_patch)

            # Clean up memory by converting sets to lists only for logging
            print(f'active: {len(active_point_ids)}, inactive: {len(inactive_point_ids)}, source: {len(source_ids_point)}')

        # Convert sets to lists only when needed for plotting
        rg_plot_finished(
            cloud=cloud,
            active_points=list(active_point_ids),
            inactive_points=list(inactive_point_ids),
            source_points=list(source_ids_point),
            sink_points_patch_ids=list(sink_ids_patch)
        )

    return cloud