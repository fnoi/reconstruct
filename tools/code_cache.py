import copy

import numpy as np
import open3d as o3d

from matplotlib import pyplot as plt

from scipy.spatial import KDTree, ConvexHull, Delaunay

import pandas as pd

from tools.visual import rg_plot_active, rg_plot_what

pd.options.mode.copy_on_write = True

from tqdm import tqdm

from tools.local import neighborhood_search, supernormal_svd_s1, consistency_flip, angular_deviation, supernormal_confidence, subset_cluster_neighbor_search


def region_growing_rev(cloud, config):
    if 'id' not in cloud.columns:
        cloud['id'] = range(len(cloud))

    cloud['instance_pr'] = 0

    # initiate seed, source, sink, active, inactive
    seed_point_id = None
    seed_patch_id = None
    source_point_ids = cloud['id'].to_list()
    source_patch_ids = cloud['ransac_patch'].unique().tolist()
    sink_point_ids = []
    sink_patch_ids = []
    active_point_ids = []
    active_patch_ids = []

    # counters
    counter_patch = 0
    game_over = False

    plot_count = 0
    max_empty_loop = 0

    # main loop, exit when done
    while True:
        if (
                len(source_point_ids) <= config.region_growing.leftover_relative * len(cloud)
                or
                game_over
        ):
            break
        counter_patch += 1
        print(f'Cluster {counter_patch}')

        #### INITIATE CLUSTER: SEED PATCH ####

        # get seed point by row value 'id'
        source_cloud = cloud[cloud['id'].isin(source_point_ids)]
        # source_cloud = cloud.loc[source_point_ids]
        # source_cloud.sort_values(by='confidence', ascending=True, inplace=True)
        source_cloud.sort_values(by='csn_confidence', ascending=True, inplace=True)
        for i, row in source_cloud.iterrows():
            if (
                    row['ransac_patch'] != 0 and
                    row['ransac_patch'] not in sink_patch_ids and
                    len(cloud[cloud['ransac_patch'] == row['ransac_patch']]) >= config.region_growing.min_patch_size
            ):
                seed_point_id = row['id']
                seed_patch_id = row['ransac_patch']
                seed_confidence = row['confidence']
                seed_sn = np.asarray(row[['snx', 'sny', 'snz']].values)
        # get all rows where ransac patch is the same, and turn the row's id into a list
        active_point_ids = source_cloud[source_cloud['ransac_patch'] == seed_patch_id]['id'].to_list()
        active_patch_ids = [seed_patch_id]
        # remove seed point from source
        source_point_ids = list(set(source_point_ids) - {seed_point_id})
        # source_point_ids.remove(seed_point_id)
        # remove seed patch from source
        source_patch_ids = list(set(source_patch_ids) - {seed_patch_id})
        # source_patch_ids.remove(seed_patch_id)

        #### CLUSTER GROWTH ####
        segment_iter = 0

        inactive_point_ids = []
        inactive_patch_ids = []

        chk_log_patches = []
        chk_log_points = []

        floating_points_dict = {}

        while True:
            segment_iter += 1
            print(f'growth iter {segment_iter}')
            # find points in bbox around active cluster
            cluster_neighbors, reduced_cloud = subset_cluster_neighbor_search(cloud, active_point_ids, config)
            ship_neighbors = list(set(cluster_neighbors))

            # ship_neighbors is a temporary fix
            if len(cluster_neighbors) == 0:
                print('no neighbors')
                game_over = True
                break

            cluster_neighbors = list(set(cluster_neighbors) - set(active_point_ids))

            if segment_iter == 1 and True == False:
                cluster_sn = cloud.loc[seed_point_id, ['snx', 'sny', 'snz']].values
                cluster_rn = cloud.loc[seed_point_id, ['rnx', 'rny', 'rnz']].values
                cluster_confidence = seed_confidence
            else:
                # _normals = np.asarray(reduced_cloud.loc[reduced_cloud['id'].isin(active_point_ids)][['nx', 'ny', 'nz']]) # TODO: ???

                __normals = np.asarray(reduced_cloud.loc[reduced_cloud['id'].isin(ship_neighbors)][['nx', 'ny', 'nz']])
                # add a check for supernormal quality to decide if to use the cluster_sn or to recalculate
                # cluster_sn = supernormal_svd(_normals)

                cluster_sn, _s1, _s2, _s3 = supernormal_svd_s1(__normals, full_return=True)
                cluster_confidence = supernormal_confidence(cluster_sn, __normals, _s1, _s2, _s3)

                # cluster_csn = cloud.loc[seed_point_id, ['csnx', 'csny', 'csnz']].values

                if seed_confidence > cluster_confidence:
                    print('seed better')
                    cluster_sn = seed_sn
                    cluster_confidence = seed_confidence
                else:
                    print('seed worse')

                # reduce cloud to active ids
                _cloud = cloud.loc[cloud['id'].isin(active_point_ids)]
                _patch_id = _cloud['ransac_patch'].mode().iloc[0]
                first_row = _cloud[_cloud['ransac_patch'] == _patch_id].iloc[0]
                cluster_rn = np.asarray(first_row[['rnx', 'rny', 'rnz']].values)

            neighbor_patch_ids = reduced_cloud.loc[reduced_cloud['id'].isin(cluster_neighbors)]['ransac_patch'].unique().tolist()
            if 0 in neighbor_patch_ids:
                neighbor_patch_ids.remove(0)
                # neighbor_patch_ids = list(set(neighbor_patch_ids)).sort()
            neighbor_unpatched_point_ids = reduced_cloud[reduced_cloud['ransac_patch'] == 0]['id'].to_list()
            # neighbor_unpatched_point_ids = list(set(neighbor_unpatched_point_ids)).sort()

            chk_neighbors_0 = chk_log_patches == len(neighbor_patch_ids)
            chk_log_patches = len(neighbor_patch_ids)
            chk_neighbors_1 = chk_log_points == len(neighbor_unpatched_point_ids)
            chk_log_points = len(neighbor_unpatched_point_ids)

            # patch-growth stopping criterion
            if chk_neighbors_0 and chk_neighbors_1:
                floating_points_dict[seed_patch_id] = neighbor_unpatched_point_ids
                sink_patch_ids.extend(active_patch_ids)

                # source_point_ids = list(set(source_point_ids) - set(active_point_ids))
                # source_patch_ids.remove(active_patch_ids)
                # source_patch_ids = list(set(source_patch_ids))

                cloud.loc[cloud['id'].isin(active_point_ids), 'instance_pr'] = counter_patch
                # plot full cloud and highlight active
                rg_plot_active(cloud, active_point_ids, counter_patch)

                break


            for neighbor_patch in neighbor_patch_ids:
                if (
                        neighbor_patch in sink_patch_ids or
                        neighbor_patch in active_patch_ids or
                        neighbor_patch in inactive_patch_ids or
                        neighbor_patch not in source_patch_ids
                ):
                    continue
                else:
                    neighbor_patch_sns = consistency_flip(
                        np.asarray(
                            cloud.loc[cloud['ransac_patch'] == neighbor_patch][['snx', 'sny', 'snz']].values
                        )
                    )
                    neighbor_patch_sn = np.mean(neighbor_patch_sns, axis=0)
                    neighbor_patch_rn = cloud.loc[cloud['ransac_patch'] == neighbor_patch].iloc[0][['rnx', 'rny', 'rnz']].values

                    neighbor_patch_first_id = cloud.loc[cloud['ransac_patch'] == neighbor_patch].iloc[0]['id']
                    neighbor_patch_csn = cloud.loc[neighbor_patch_first_id, ['csnx', 'csny', 'csnz']].values

                    deviation_sn_old = angular_deviation(cluster_sn, neighbor_patch_sn) % 180
                    deviation_sn_old = min(deviation_sn_old, 180 - deviation_sn_old)

                    deviation_sn = angular_deviation(cluster_sn, neighbor_patch_csn) % 180
                    deviation_sn = min(deviation_sn, 180 - deviation_sn)

                    deviation_rn = angular_deviation(cluster_rn, neighbor_patch_rn) % 90
                    deviation_rn = min(deviation_rn, 90 - deviation_rn)

                    chk_add_patch_0 = deviation_sn < config.region_growing.supernormal_angle_deviation_patch
                    chk_add_patch_1 = deviation_rn < config.region_growing.ransacnormal_angle_deviation_patch

                    if chk_add_patch_0 and chk_add_patch_1:
                        color = 'g'
                        point_ids = cloud[cloud['ransac_patch'] == neighbor_patch]['id'].to_list()
                        active_plot = copy.deepcopy(active_point_ids)
                        active_point_ids.extend(point_ids)
                        active_patch_ids.append(neighbor_patch)
                        source_point_ids = list(set(source_point_ids) - set(active_point_ids))
                        # srouce_point_ids =
                        source_patch_ids.remove(neighbor_patch)
                        source_patch_ids = list(set(source_patch_ids))
                    else:
                        color = 'r'
                        point_ids = cloud[cloud['ransac_patch'] == neighbor_patch]['id'].to_list()
                        active_plot = copy.deepcopy(active_point_ids)
                        inactive_point_ids.extend(point_ids)
                        inactive_patch_ids.append(neighbor_patch)

                    plot_all = False
                    if plot_all:
                        rg_plot_what(cloud=cloud, active_point_ids=active_point_ids,
                                     point_ids=point_ids, cluster_sn=cluster_sn,
                                     cluster_rn=cluster_rn, neighbor_patch_sn=neighbor_patch_sn,
                                     neighbor_patch_rn=neighbor_patch_rn, neighbor_patch_csn=neighbor_patch_csn,
                                     deviation_sn=deviation_sn, deviation_sn_old=deviation_sn_old,
                                     deviation_rn=deviation_rn, cluster_confidence=cluster_confidence,
                                     plot_count=plot_count, config=config)

                    print(f'active: {len(active_point_ids)}, inactive: {len(inactive_point_ids)}, source: {len(source_point_ids)}')

    return cloud


