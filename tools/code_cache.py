import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from tqdm import tqdm

from tools.local import neighborhood_search, supernormal_svd, consistency_flip, angular_deviation


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
            if row['ransac_patch'] != 0:
                seed_point_id = row['id']
                seed_patch_id = row['ransac_patch']
        # get all rows where ransac patch is the same, and turn the row's id into a list
        active_point_ids = source_cloud[source_cloud['ransac_patch'] == seed_patch_id]['id'].to_list()
        active_patch_ids = [seed_patch_id]
        # remove seed point from source
        source_point_ids.remove(seed_point_id)
        # remove seed patch from source
        source_patch_ids.remove(seed_patch_id)

        #### CLUSTER GROWTH ####
        segment_iter = 0
        active_unpatched_point_ids = []
        x_l_a = 0
        x_l_i = 0
        x_l_s = 0
        log_unpatched = None
        new_segment = False
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
            else:
                cluster_sn = supernormal_svd(
                    np.asarray(reduced_cloud.loc[reduced_cloud['id'].isin(active_point_ids)][['nx', 'ny', 'nz']])
                )
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

            if chk_neighbors_0 and chk_neighbors_1:
                floating_points_dict[seed_patch_id] = neighbor_unpatched_point_ids
                break

            try:
                print(len(neighbor_unpatched_point_ids))
            except:
                a = 0


            for neighbor_patch in neighbor_patch_ids:
                if (
                        neighbor_patch in sink_patch_ids or
                        neighbor_patch in active_patch_ids or
                        neighbor_patch in inactive_patch_ids
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
                        active_point_ids.extend(point_ids)
                        active_patch_ids.append(neighbor_patch)
                        source_point_ids = list(set(source_point_ids) - set(point_ids))
                        source_patch_ids.remove(neighbor_patch)
                        source_patch_ids = list(set(source_patch_ids))
                    else:
                        color = 'r'
                        point_ids = cloud[cloud['ransac_patch'] == neighbor_patch]['id'].to_list()
                        inactive_point_ids.extend(point_ids)
                        inactive_patch_ids.append(neighbor_patch)


                    if True:
                        fig = plt.figure(figsize=(18.5, 10.5))

                        # First subplot
                        ax1 = fig.add_subplot(121, projection='3d')
                        rest_idx = list(set(cloud['id'].to_list()) - set(active_point_ids) - set(point_ids))
                        ax1.scatter(cloud.loc[active_point_ids, 'x'], cloud.loc[active_point_ids, 'y'], cloud.loc[active_point_ids, 'z'], c='b', s=0.2)
                        ax1.scatter(cloud.loc[point_ids, 'x'], cloud.loc[point_ids, 'y'], cloud.loc[point_ids, 'z'], c=color, s=0.2)
                        ax1.scatter(cloud.loc[rest_idx, 'x'], cloud.loc[rest_idx, 'y'], cloud.loc[rest_idx, 'z'], c='grey', s=0.1, alpha=0.2)
                        ax1.set_title("Original Perspective")

                        # Second subplot with rotated perspective
                        ax2 = fig.add_subplot(122, projection='3d')
                        ax2.scatter(cloud.loc[active_point_ids, 'x'], cloud.loc[active_point_ids, 'y'], cloud.loc[active_point_ids, 'z'], c='b', s=0.2)
                        ax2.scatter(cloud.loc[point_ids, 'x'], cloud.loc[point_ids, 'y'], cloud.loc[point_ids, 'z'], c=color, s=0.2)
                        ax2.scatter(cloud.loc[rest_idx, 'x'], cloud.loc[rest_idx, 'y'], cloud.loc[rest_idx, 'z'], c='grey', s=0.1, alpha=0.2)
                        ax2.view_init(elev=90, azim=45)  # Adjust these angles for desired perspective
                        ax2.set_title("Rotated Perspective (from above)")

                        plt.tight_layout()
                        plt.show()

                    print(f'active: {len(active_point_ids)}, inactive: {len(inactive_point_ids)}, source: {len(source_point_ids)}')



                    a = 0


            # # check if neighbors can be added to active cluster
            # for check_neighbor_point_id in cluster_neighbors:
            #     print('checking neighbors')
            #     check_neighbor_patch_id = cloud.loc[check_neighbor_point_id, 'ransac_patch']
            #     if (
            #             check_neighbor_patch_id in sink_patch_ids or
            #             check_neighbor_patch_id in active_patch_ids or
            #             check_neighbor_patch_id in inactive_patch_ids
            #     ):
            #         continue
            #
            #     elif check_neighbor_patch_id == 0:
            #         active_unpatched_point_ids.append(check_neighbor_point_id)
            #         active_unpatched_point_ids = list(set(active_unpatched_point_ids))
            #         if log_unpatched == active_unpatched_point_ids:
            #             new_segment = True
            #         log_unpatched = active_unpatched_point_ids
            #     else:
            #
            #         neighbor_point_sn = cloud.loc[check_neighbor_point_id, ['snx', 'sny', 'snz']].values
            #         neighbor_point_rn = cloud.loc[check_neighbor_point_id, ['rnx', 'rny', 'rnz']].values
            #         neighbor_patch_sn = np.mean(
            #             np.asarray(cloud.loc[cloud['ransac_patch'] == check_neighbor_patch_id][['snx', 'sny', 'snz']]),
            #             axis=0)
            #         # neighbor_point_rn = neighbor_patch_rn
            #         neighbor_patch_ids = cloud[cloud['ransac_patch'] == check_neighbor_patch_id]['id'].to_list()
            #
            #         deviation_sn = angular_deviation(cluster_sn, neighbor_patch_sn) % 180
            #         deviation_sn = min(deviation_sn, 180 - deviation_sn)
            #
            #         deviation_rn = angular_deviation(cluster_rn, neighbor_point_rn) % 90
            #         deviation_rn = min(deviation_rn, 90 - deviation_rn)
            #
            #         check_0 = deviation_sn < config.region_growing.supernormal_angle_deviation_patch
            #         check_1 = deviation_rn < config.region_growing.ransacnormal_angle_deviation_patch
            #
            #         if check_0 and check_1 and not new_segment:
            #             color = 'g'
            #             active_point_ids.append(check_neighbor_point_id)
            #             active_point_ids = list(set(active_point_ids))
            #             active_patch_ids.append(check_neighbor_patch_id)
            #             active_patch_ids = list(set(active_patch_ids))
            #             source_point_ids.remove(check_neighbor_point_id)
            #             source_point_ids = list(set(source_point_ids))
            #             source_patch_ids.remove(check_neighbor_patch_id)
            #             source_patch_ids = list(set(source_patch_ids))
            #
            #         else:
            #             color = 'r'
            #             inactive_point_ids.append(check_neighbor_point_id)
            #             inactive_point_ids = list(set(inactive_point_ids))
            #             inactive_patch_ids.append(check_neighbor_patch_id)
            #             inactive_patch_ids = list(set(inactive_patch_ids))
            #
            #         if True:
            #             # scatter plot of xyz values for active point ids and check neighbor point id
            #             fig = plt.figure()
            #             ax = fig.add_subplot(111, projection='3d')
            #             # point ids that are not active_point_ids or neighbor_patch_ids
            #             rest_idx = list(set(cloud['id'].to_list()) - set(active_point_ids) - set(neighbor_patch_ids))
            #             ax.scatter(cloud.loc[active_point_ids, 'x'], cloud.loc[active_point_ids, 'y'], cloud.loc[active_point_ids, 'z'], c='b', s=0.2)
            #             ax.scatter(cloud.loc[neighbor_patch_ids, 'x'], cloud.loc[neighbor_patch_ids, 'y'], cloud.loc[neighbor_patch_ids, 'z'], c=color, s=0.2)
            #             ax.scatter(cloud.loc[rest_idx, 'x'], cloud.loc[rest_idx, 'y'], cloud.loc[rest_idx, 'z'], c='grey', s=0.1, alpha=0.2)
            #             fig.set_size_inches(18.5, 10.5)
            #             plt.show()
            #
            #         check_01 = len(active_point_ids) == x_l_a
            #         check_02 = len(inactive_point_ids) == x_l_i
            #         check_03 = len(source_point_ids) == x_l_s
            #         if check_01 and check_02 and check_03:
            #             break
            #         else:
            #             x_l_a = len(active_point_ids)
            #             x_l_i = len(inactive_point_ids)
            #             x_l_s = len(source_point_ids)
            #
            # if new_segment:
            #     break
            #
            #
            #
            #         # am ende des loops: add to sink, clear active
            #
            #
            #
            #



        # check neighborhood of seed point

        # in the neighborhood check which points are ok, and part of planar patches
        # store points that are not part of patches

        # repeat neighborhood checks until no change

        # all the non-patch points are now checked, if they can be added based on separate rules/ parameters

        # now initiate new cluster with seed, empty burnt, active, update sink accordingly
