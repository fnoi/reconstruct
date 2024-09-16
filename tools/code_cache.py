import copy

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

import pandas as pd
pd.options.mode.copy_on_write = True

from tqdm import tqdm

from tools.local import neighborhood_search, supernormal_svd, consistency_flip, angular_deviation, supernormal_confidence


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
    inactive_point_ids = []
    inactive_patch_ids = []

    # counters
    counter_patch = 0

    # main loop, exit when done
    while True:
        if len(source_point_ids) <= config.region_growing.leftover_relative * len(cloud):
            break
        counter_patch += 1
        print(f'Cluster {counter_patch}')

        #### INITIATE CLUSTER: SEED PATCH ####

        # get seed point by row value 'id'
        source_cloud = cloud[cloud['id'].isin(source_point_ids)]
        # source_cloud = cloud.loc[source_point_ids]
        source_cloud.sort_values(by='confidence', ascending=False, inplace=True)
        for i, row in source_cloud.iterrows():
            if (
                    row['ransac_patch'] != 0 and
                    row['ransac_patch'] not in sink_patch_ids and
                    row['ransac_patch']
            ):
                seed_point_id = row['id']
                seed_patch_id = row['ransac_patch']
                seed_confidence = row['confidence']
                seed_sn = np.asarray(row[['snx', 'sny', 'snz']].values)
        # get all rows where ransac patch is the same, and turn the row's id into a list
        active_point_ids = source_cloud[source_cloud['ransac_patch'] == seed_patch_id]['id'].to_list()
        active_patch_ids = [seed_patch_id]
        # remove seed point from source
        source_point_ids.remove(seed_point_id)
        # remove seed patch from source
        source_patch_ids.remove(seed_patch_id)

        #### CLUSTER GROWTH ####
        segment_iter = 0

        chk_log_patches = []
        chk_log_points = []

        floating_points_dict = {}

        while True:
            segment_iter += 1
            print(f'growth iter {segment_iter}')
            # find points in bbox around active cluster
            active_limits = [np.min(cloud.loc[active_point_ids, 'x']),
                             np.max(cloud.loc[active_point_ids, 'x']),
                             np.min(cloud.loc[active_point_ids, 'y']),
                             np.max(cloud.loc[active_point_ids, 'y']),
                             np.min(cloud.loc[active_point_ids, 'z']),
                             np.max(cloud.loc[active_point_ids, 'z'])]
            neighbor_ids_potential = neighborhood_search(
                cloud=cloud, seed_id=None, config=config,
                step='bbox_mask', cluster_lims=active_limits
            )
            reduced_cloud = cloud.loc[cloud['id'].isin(neighbor_ids_potential)]
            reduced_tree = KDTree(reduced_cloud[['x', 'y', 'z']].values)
            cluster_neighbors = []
            for x_active_point in active_point_ids:
                cluster_neighbors.extend(neighborhood_search(
                    cloud=reduced_cloud, seed_id=x_active_point, config=config, cloud_tree=reduced_tree,
                    step='patch growing', patch_sn=None, cluster_lims=None
                ))
            cluster_neighbors = list(set(cluster_neighbors))
            cluster_neighbors = reduced_cloud.iloc[cluster_neighbors]['id'].to_list()
            cluster_neighbors = list(set(cluster_neighbors) - set(active_point_ids))

            if segment_iter == 1:
                cluster_sn = cloud.loc[seed_point_id, ['snx', 'sny', 'snz']].values
                cluster_rn = cloud.loc[seed_point_id, ['rnx', 'rny', 'rnz']].values
                cluster_confidence = seed_confidence
            else:
                _normals = np.asarray(reduced_cloud.loc[reduced_cloud['id'].isin(active_point_ids)][['nx', 'ny', 'nz']])
                # add a check for supernormal quality to decide if to use the cluster_sn or to recalculate
                # cluster_sn = supernormal_svd(_normals)

                cluster_sn, _s1, _s2, _s3 = supernormal_svd(_normals, full_return=True)
                cluster_confidence = supernormal_confidence(cluster_sn, _normals, _s1, _s2, _s3)

                if seed_confidence > cluster_confidence:
                    cluster_sn = seed_sn
                    cluster_confidence = seed_confidence

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
                break

            try:
                print(len(neighbor_unpatched_point_ids))
            except:
                a = 0


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

                    deviation_sn = angular_deviation(cluster_sn, neighbor_patch_sn) % 180
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
                        source_point_ids = list(set(source_point_ids) - set(point_ids))
                        source_patch_ids.remove(neighbor_patch)
                        source_patch_ids = list(set(source_patch_ids))
                    else:
                        color = 'r'
                        point_ids = cloud[cloud['ransac_patch'] == neighbor_patch]['id'].to_list()
                        active_plot = copy.deepcopy(active_point_ids)
                        inactive_point_ids.extend(point_ids)
                        inactive_patch_ids.append(neighbor_patch)


                    if True:
                        fig = plt.figure(figsize=(20, 20))

                        # Function to create scatter plot with supernormal vector line
                        def create_scatter(ax, view_elev, view_azim):
                            rest_idx = list(set(cloud['id'].to_list()) - set(active_plot) - set(point_ids))
                            ax.scatter(cloud.loc[active_plot, 'x'], cloud.loc[active_plot, 'y'], cloud.loc[active_plot, 'z'], c='b', s=0.2)
                            ax.scatter(cloud.loc[point_ids, 'x'], cloud.loc[point_ids, 'y'], cloud.loc[point_ids, 'z'], c=color, s=0.2)
                            ax.scatter(cloud.loc[rest_idx, 'x'], cloud.loc[rest_idx, 'y'], cloud.loc[rest_idx, 'z'], c='grey', s=0.1, alpha=0.2)

                            # Calculate centroid of active points
                            centroid = cloud.loc[active_point_ids, ['x', 'y', 'z']].mean().values

                            # Create line representing cluster_sn
                            end_point = centroid + .5 * cluster_sn
                            ax.plot([centroid[0], end_point[0]],
                                    [centroid[1], end_point[1]],
                                    [centroid[2], end_point[2]],
                                    color='b', linewidth=2)

                            # Create line representing cluster_rn, dashed
                            end_point = centroid + .5 * cluster_rn
                            ax.plot([centroid[0], end_point[0]],
                                    [centroid[1], end_point[1]],
                                    [centroid[2], end_point[2]],
                                    color='b', linewidth=2, linestyle='dashed')

                            # create line representing neighbor_patch_sn
                            end_point = centroid + .5 * neighbor_patch_sn
                            ax.plot([centroid[0], end_point[0]],
                                    [centroid[1], end_point[1]],
                                    [centroid[2], end_point[2]],
                                    color=color, linewidth=2)

                            # create line representing neighbor_patch_rn, dashed
                            end_point = centroid + .5 * neighbor_patch_rn
                            ax.plot([centroid[0], end_point[0]],
                                    [centroid[1], end_point[1]],
                                    [centroid[2], end_point[2]],
                                    color=color, linewidth=2, linestyle='dashed')

                            ax.view_init(elev=view_elev, azim=view_azim)

                        # Subplot 1: Original perspective
                        ax1 = fig.add_subplot(221, projection='3d')
                        create_scatter(ax1, 20, 45)

                        # Subplot 2: Top-down view
                        ax2 = fig.add_subplot(222, projection='3d')
                        create_scatter(ax2, 90, 0)

                        # Subplot 3: Side view (YZ plane)
                        ax3 = fig.add_subplot(223, projection='3d')
                        create_scatter(ax3, 0, 0)

                        # Subplot 4: Front view (XZ plane)
                        ax4 = fig.add_subplot(224, projection='3d')
                        create_scatter(ax4, 0, 90)

                        fig.suptitle(f'supernormal (sn) deviation: {deviation_sn:.2f}°\n'
                                     f'ransac normal (rn) deviation: {deviation_rn:.2f}°\n'
                                     f'cluster confidence: {cluster_confidence:.2f}\n')
                        plt.tight_layout()
                        plt.show()

                    print(f'active: {len(active_point_ids)}, inactive: {len(inactive_point_ids)}, source: {len(source_point_ids)}')


