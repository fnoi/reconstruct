import copy
from time import perf_counter

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

    return should_add, deviation_sn


def region_growing_main(cloud, config):
    """Memory-optimized main region growing function with source point reduction"""
    # Keep original cloud for final labeling
    original_cloud = cloud.copy()

    if 'id' not in cloud.columns:
        cloud['id'] = range(len(cloud))
        original_cloud['id'] = range(len(original_cloud))

    cloud['instance_pr'] = 0
    original_cloud['instance_pr'] = 0

    # Use sets for operations but convert to list for pandas indexing
    source_ids_point = set(cloud['id'])
    source_ids_patch = set(cloud['ransac_patch'].unique()) - {0}
    sink_ids_patch = set()

    counter_patch = 0
    end_threshold = int(config.region_growing.leftover_relative * len(cloud))

    print(f'ending threshold: {end_threshold}')

    while len(source_ids_point) > end_threshold:
        counter_patch += 1
        timer = perf_counter()

        # Get seed point from reduced cloud
        seed_result = get_seed_point(cloud, list(source_ids_point), sink_ids_patch, config.region_growing.min_patch_size)
        if seed_result is None:
            break
        seed_id_point, seed_id_patch, seed_confidence, seed_sn = seed_result
        if seed_confidence is None:
            break

        # Check if seed point has high confidence
        high_confidence_mode = seed_confidence > 5
        if high_confidence_mode:
            print(f"High confidence seed point (confidence: {seed_confidence:.2f})")

        # Initialize active sets
        active_point_ids = set(cloud[cloud['ransac_patch'] == seed_id_patch]['id'])
        active_patch_ids = {seed_id_patch}

        # Update source sets efficiently
        source_ids_point -= active_point_ids
        source_ids_patch.remove(seed_id_patch)

        # Growth phase
        segment_iter = 0
        prev_patches_count = 0
        prev_points_count = 0

        while True:
            segment_iter += 1

            # Find neighbors using reduced cloud
            neighbor_ids_cluster, neighbor_cloud_cluster = subset_cluster_neighbor_search(
                cloud, list(active_point_ids), config)

            if not neighbor_ids_cluster:
                break

            neighbor_ids_cluster = set(neighbor_ids_cluster) - active_point_ids
            if not neighbor_ids_cluster:
                break

            # Calculate normals
            cluster_sn, cluster_rn, cluster_conf = calculate_cluster_normals(
                cloud, neighbor_cloud_cluster, list(neighbor_ids_cluster),
                seed_id_point, seed_sn, seed_confidence, list(active_point_ids))

            # Process neighbors efficiently
            neighbor_ids_patch = set(neighbor_cloud_cluster.loc[
                                         neighbor_cloud_cluster['id'].isin(list(neighbor_ids_cluster))
                                     ]['ransac_patch'].unique()) - {0}

            neighbor_ids_points_unpatched = set(
                neighbor_cloud_cluster[neighbor_cloud_cluster['ransac_patch'] == 0]['id']
            )

            # Check for growth completion
            if (len(neighbor_ids_patch) == prev_patches_count and
                    len(neighbor_ids_points_unpatched) == prev_points_count):
                break

            prev_patches_count = len(neighbor_ids_patch)
            prev_points_count = len(neighbor_ids_points_unpatched)

            # Evaluate neighbor patches efficiently
            for neighbor_patch in neighbor_ids_patch:
                if (neighbor_patch in sink_ids_patch or
                        neighbor_patch in active_patch_ids or
                        neighbor_patch not in source_ids_patch):
                    continue

                should_add, deviation_sn = evaluate_neighbor_patch(
                    cloud, neighbor_patch, cluster_sn, cluster_rn, config, high_confidence_mode)

                if should_add:
                    point_ids = set(cloud[cloud['ransac_patch'] == neighbor_patch]['id'])
                    active_point_ids.update(point_ids)
                    active_patch_ids.add(neighbor_patch)
                    source_ids_point -= point_ids
                    source_ids_patch.remove(neighbor_patch)

        # Label points in both clouds
        cloud.loc[list(active_point_ids), 'instance_pr'] = counter_patch
        original_cloud.loc[list(active_point_ids), 'instance_pr'] = counter_patch
        sink_ids_patch.update(active_patch_ids)

        # Reduce cloud to only source points for next iteration
        cloud = cloud[cloud['id'].isin(source_ids_point)].copy()

        elapsed = round(perf_counter() - timer, 2)
        elapsed_rel = round(len(active_point_ids) / elapsed, 2)
        print(f'cluster {counter_patch} finished after {elapsed}s with {len(active_point_ids)} points at {elapsed_rel}p/s    ::'
              f'source: {len(source_ids_point)}')

    return original_cloud