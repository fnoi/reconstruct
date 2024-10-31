import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from tools.local import neighborhood_search, angular_deviation, supernormal_svd_s1


def region_growing(cloud, config):
    """region growing algorithm for instance segmentation"""
    # check if cloud has 'id' column
    if 'id' not in cloud.columns:
        # add column to cloud that stores integer ID
        cloud['id'] = [i for i in range(len(cloud))]

    # actual annotation
    label_instance = 1
    cloud['instance_pr'] = 0  # unlabeled
    # source: unsegmented
    source_point_ids = cloud['id'].tolist()
    source_patch_ids = cloud['ransac_patch'].unique().tolist()
    # active: now growing, assigned to current segment
    active_point_ids = []
    active_patch_ids = []
    # sink: already segmented
    sink_point_ids = []
    sink_patch_ids = []

    pointcount = 0
    patchcount = 0
    while True:
        pointcount += 1
        print(f'segment counter: {pointcount}')
        if len(source_point_ids) <= config.region_growing.leftover_relative * len(cloud):
            break
        # find seed point
        source_cloud = cloud.loc[source_point_ids]
        # sort source cloud by confidence
        source_cloud.sort_values(by='confidence', ascending=False, inplace=True)
        # seed point
        for i, row in source_cloud.iterrows():
            if row['ransac_patch'] == 0:
                continue
            else:
                seed_point_id = row['id']
                break

        # retrieve 'sn' and 'rn' values from the row where 'id' == seed_point_id
        seed_sn = cloud.loc[cloud['id'] == seed_point_id, ['snx', 'sny', 'snz']].values
        seed_rn = cloud.loc[cloud['id'] == seed_point_id, ['rnx', 'rny', 'rnz']].values

        # seed patch
        seed_patch_id = cloud.loc[cloud['id'] == seed_point_id, 'ransac_patch'].values[0]
        # find 'id's of points in the seed patch
        seed_patch_point_ids = cloud.loc[cloud['ransac_patch'] == seed_patch_id, 'id'].tolist()

        # inventory
        active_point_ids.extend(seed_patch_point_ids)
        seed_patch_point_ids_set = set(seed_patch_point_ids)
        source_point_ids = [_ for _ in source_point_ids if _ not in seed_patch_point_ids_set]

        active_patch_ids.append(cloud.loc[cloud['id'] == seed_point_id, 'ransac_patch'].values[0])
        print(seed_patch_id)
        if seed_patch_id in source_patch_ids:
            source_patch_ids.remove(seed_patch_id)
        else:
            break

        # active_patch_ids.append(cloud.loc[seed_point_id, 'ransac_patch'])
        # source_patch_ids.remove(cloud.loc[seed_point_id, 'ransac_patch'])

        cluster_blacklist = []
        growth_iter = 0
        len_log = 0

        while True:
            patchcount += 1
            print(f'iteration in segment: {patchcount}')
            growth_iter += 1
            # candidate_point_ids = [_ for _ in active_point_ids
            #                        if _ not in sink_point_ids
            #                        and _ not in cluster_blacklist]
            active_limits = [np.min(cloud.loc[active_point_ids, 'x']),
                             np.max(cloud.loc[active_point_ids, 'x']),
                             np.min(cloud.loc[active_point_ids, 'y']),
                             np.max(cloud.loc[active_point_ids, 'y']),
                             np.min(cloud.loc[active_point_ids, 'z']),
                             np.max(cloud.loc[active_point_ids, 'z'])]
            potential_neighbors = neighborhood_search(
                cloud=cloud, seed_id=None, config=config,
                step='bbox_mask', cluster_lims=active_limits
            )
            potential_cloud = cloud.loc[cloud['id'].isin(potential_neighbors)]

            # scatter plot potential cloud within limits
            _plot_cluster = cloud.loc[cloud['id'].isin(active_point_ids)]
            _plot_neighbors = cloud.loc[cloud['id'].isin(potential_neighbors)]
            _plot_idle_cluster = cloud.loc[~cloud['id'].isin(active_point_ids)]
            _plot_idle_neighbors = cloud.loc[~cloud['id'].isin(potential_neighbors)]

            # fig = plt.figure()
            # ax1 = fig.add_subplot(121, projection='3d')
            # ax2 = fig.add_subplot(122, projection='3d')
            # ax1.scatter(_plot_cluster['x'], _plot_cluster['y'], _plot_cluster['z'], s=.02, c='violet')
            # ax1.scatter(_plot_idle_cluster['x'], _plot_idle_cluster['y'], _plot_idle_cluster['z'], s=.02, c='grey', alpha=0.4)
            # ax2.scatter(_plot_neighbors['x'], _plot_neighbors['y'], _plot_neighbors['z'], s=.02, c='green')
            # ax2.scatter(_plot_idle_neighbors['x'], _plot_idle_neighbors['y'], _plot_idle_neighbors['z'], s=.02, c='grey', alpha=0.4)
            # ax1.set_aspect('equal')
            # ax2.set_aspect('equal')
            # ax1.axis('off')
            # ax2.axis('off')
            # plt.tight_layout()
            # plt.title(f'potential neighbors: {len(potential_neighbors)}')
            # plt.show()

            actual_neighbors = []
            smart_choices = True
            if smart_choices and growth_iter > 1:
                # calculate supernormal based on cluster points
                # cluster_normals = cloud.loc[cloud['id'].isin(active_point_ids), ['nx', 'ny', 'nz']].to_numpy()
                # cluster_normals = potential_cloud.loc[potential_cloud['id'].isin(active_point_ids), ['rnx', 'rny', 'rnz']].to_numpy()
                cluster_normals = potential_cloud.loc[potential_cloud['id'].isin(active_point_ids), ['nx', 'ny', 'nz']].to_numpy()
                # cluster_normals = cloud.loc[active_point_ids, ['nx', 'ny', 'nz']].to_numpy()
                print('in')
                cluster_sn = supernormal_svd_s1(cluster_normals)
                print('out')
                cluster_sn /= np.linalg.norm(cluster_sn)
                # find the most confident % of cluster points per config
                # cluster_confidences = cloud.loc[active_point_ids, 'confidence']
                # cluster_confidences.sort_values(ascending=False, inplace=True)
                # cluster_confidences = cluster_confidences.iloc[:int(len(cluster_confidences) * config.region_growing.supernormal_vote_of_confidence)]
                # cluster_confidence_ids = cluster_confidences.index.tolist()
                #
                # cluster_sn = cloud.loc[cluster_confidence_ids, ['snx', 'sny', 'snz']].to_numpy()
                # cluster_sn_sum = np.sum(cluster_sn, axis=0)
                # cluster_sn = cluster_sn_sum / np.linalg.norm(cluster_sn_sum)
                # # cluster_sn /= np.linalg.norm(cluster_sn)

                # find the biggest patch in the cluster
                ransac_patch_sizes = cloud.loc[cloud['id'].isin(active_point_ids), 'ransac_patch'].value_counts()
                # ransac_patch_sizes = cloud.loc[active_point_ids, 'ransac_patch'].value_counts()
                biggest_patch_id = ransac_patch_sizes.idxmax()

                if biggest_patch_id == 0:
                    cluster_rn = seed_rn
                else:
                    cluster_rn = cloud.loc[cloud['ransac_patch'] == biggest_patch_id, ['rnx', 'rny', 'rnz']][:1]
                    cluster_rn = cluster_rn.values[0]
                    cluster_rn /= np.linalg.norm(cluster_rn)
            else:
                cluster_sn = seed_sn
                cluster_rn = seed_rn

            potential_cloud_tree = KDTree(potential_cloud[['x', 'y', 'z']].values)  # unused if not sphere...
            for active_point in tqdm(active_point_ids, desc="checking potential neighbors", total=len(active_point_ids)):
                actual_neighbors.extend(neighborhood_search(
                    cloud=potential_cloud, seed_id=active_point, config=config, cloud_tree=potential_cloud_tree,
                    step='patch growing', patch_sn=cluster_sn, cluster_lims=active_limits
                ))
            actual_neighbors = list(set(actual_neighbors))
            actual_neighbors = potential_cloud.iloc[actual_neighbors]['id'].tolist()
            actual_neighbors = [_ for _ in actual_neighbors if _ not in active_point_ids]

            # scatter plot potential cloud within limits
            _plot_cluster = cloud.loc[cloud['id'].isin(active_point_ids)]
            _plot_neighbors = cloud.loc[cloud['id'].isin(actual_neighbors)]
            _plot_idle_cluster = cloud.loc[~cloud['id'].isin(active_point_ids)]
            _plot_idle_neighbors = cloud.loc[~cloud['id'].isin(actual_neighbors)]

            # fig = plt.figure()
            # ax1 = fig.add_subplot(121, projection='3d')
            # ax2 = fig.add_subplot(122, projection='3d')
            # ax1.scatter(_plot_cluster['x'], _plot_cluster['y'], _plot_cluster['z'], s=.02, c='violet')
            # ax1.scatter(_plot_idle_cluster['x'], _plot_idle_cluster['y'], _plot_idle_cluster['z'], s=.02, c='grey', alpha=0.4)
            # ax2.scatter(_plot_neighbors['x'], _plot_neighbors['y'], _plot_neighbors['z'], s=.02, c='green')
            # ax2.scatter(_plot_idle_neighbors['x'], _plot_idle_neighbors['y'], _plot_idle_neighbors['z'], s=.02, c='grey', alpha=0.4)
            # #line plot for cluster_sn next to scatter
            # # line starts at cluster center point + 0.5 x and y
            # start_pt = np.mean(potential_cloud.loc[potential_cloud['id'].isin(active_point_ids), ['x', 'y', 'z']].values, axis=0)
            # start_pt[0:2] += 0.5
            # end_pt = (start_pt + cluster_sn).flatten()
            # ax1.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], [start_pt[2], end_pt[2]], c='orange', linewidth=2)
            # ax2.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], [start_pt[2], end_pt[2]], c='orange', linewidth=2)
            #
            # ax1.set_aspect('equal')
            # ax2.set_aspect('equal')
            # ax1.axis('off')
            # ax2.axis('off')
            # plt.title(f'actual neighbors: {len(actual_neighbors)}')
            # plt.tight_layout()
            # plt.show()

            visited_neighbors = []
            # for each of those neighbors, if they belong to a patch, check if patch can be added
            for neighbor in actual_neighbors:
                # check if candidate is available
                if (neighbor in visited_neighbors
                        or cloud[cloud['id'] == neighbor]['ransac_patch'].values[0] in active_patch_ids
                        # or cloud.loc[neighbor, 'ransac_patch'] in active_patch_ids
                        or neighbor in sink_point_ids
                ):
                    continue

                # main check
                else:
                    neighbor_patch = potential_cloud.loc[potential_cloud['id'] == neighbor, 'ransac_patch'].values[0]
                    if neighbor_patch == 0:  # free point: not member of a ransac patch
                        # check if neighbor point can be added
                        neighbor_sn = potential_cloud.loc[potential_cloud['id'] == neighbor, ['snx', 'sny', 'snz']].values
                        deviation_sn = angular_deviation(cluster_sn, neighbor_sn) % 180
                        deviation_sn = min(deviation_sn, 180 - deviation_sn)
                        # USED PARAMETER: REGION_GROWING.SUPERNORMAL_POINT_ANGLE_DEVIATION
                        if deviation_sn < config.region_growing.supernormal_angle_deviation_point:
                            active_point_ids.append(neighbor)
                            source_point_ids.remove(neighbor)
                            print(f'added point {neighbor}')
                    else:  # patch point: member of a ransac patch
                        # check if the entire patch can be added
                        neighbor_patch_sns = potential_cloud.loc[potential_cloud['ransac_patch'] == neighbor_patch, ['snx', 'sny', 'snz']].values
                        # find mean direction of sns vectors in the patch
                        neighbor_patch_sn = np.mean(neighbor_patch_sns, axis=0)
                        # neighbor_patch_sn = potential_cloud.loc[potential_cloud['id'] == neighbor, ['snx', 'sny', 'snz']].values
                        # what is the patch supernormal?!? mean SN of all points in the patch?
                        neighbor_patch_rn = potential_cloud.loc[potential_cloud['id'] == neighbor, ['rnx', 'rny', 'rnz']].values

                        deviation_sn = angular_deviation(cluster_sn, neighbor_patch_sn) % 180
                        deviation_sn = min(deviation_sn, 180 - deviation_sn)
                        deviation_rn = angular_deviation(cluster_rn, neighbor_patch_rn) % 90
                        deviation_rn = min(deviation_rn, 90 - deviation_rn)

                        # USED PARAMETER: REGION_GROWING.SUPERNORMAL_PATCH_ANGLE_DEVIATION
                        # USED PARAMETER: REGION_GROWING.RANSACNORMAL_PATCH_ANGLE_DEVIATION
                        if deviation_sn < config.region_growing.supernormal_angle_deviation_patch and \
                                deviation_rn < config.region_growing.ransacnormal_angle_deviation_patch:
                            active_patch_ids.append(neighbor_patch)
                            active_point_ids.extend(cloud[cloud['ransac_patch'] == neighbor_patch]['id'].tolist())
                            visited_neighbors.extend(cloud[cloud['ransac_patch'] == neighbor_patch]['id'].tolist())

                            if neighbor_patch not in source_patch_ids:
                                raise ValueError('patch not in source patches')
                            source_patch_ids.remove(neighbor_patch)

                            # speed up the removal of visited neighbors
                            source_point_ids_array = np.array(source_point_ids)
                            cloud_ids_array = np.array(cloud.loc[cloud['ransac_patch'] == neighbor_patch, 'id'])

                            indices_to_keep = np.where(~np.isin(source_point_ids_array, cloud_ids_array))

                            source_point_ids_filtered = source_point_ids_array[indices_to_keep]

                            source_point_ids = source_point_ids_filtered.tolist()

                            print(f'added patch {neighbor_patch}')

                        else:
                            visited_neighbors.extend(cloud[cloud['ransac_patch'] == neighbor_patch]['id'].tolist())

            plot_cluster = True
            if plot_cluster:
                # scatter plot active cloud
                fig = plt.figure()
                active_points_set = set(active_point_ids)
                inactive_point_ids = [_ for _ in cloud['id'].tolist() if _ not in active_points_set]
                ax = fig.add_subplot(111, projection='3d')
                # 1. full cloud
                ax.scatter(cloud.loc[cloud['id'].isin(inactive_point_ids), 'x'],
                           cloud.loc[cloud['id'].isin(inactive_point_ids), 'y'],
                           cloud.loc[cloud['id'].isin(inactive_point_ids), 'z'],
                           s=1, c='grey', alpha=0.4)
                # 2. active cloud
                ax.scatter(cloud.loc[cloud['id'].isin(active_point_ids), 'x'],
                           cloud.loc[cloud['id'].isin(active_point_ids), 'y'],
                           cloud.loc[cloud['id'].isin(active_point_ids), 'z'],
                           s=4, c='violet')
                ax.set_aspect('equal')
                plt.show()

            if len_log == len(active_point_ids):

                cloud.loc[active_point_ids, 'instance_pr'] = label_instance
                label_instance += 1
                # source_point_ids = [_ for _ in source_point_ids if _ not in active_point_ids]
                # source_patch_ids = [_ for _ in source_patch_ids if _ not in active_patch_ids]
                sink_point_ids.extend(active_point_ids)
                sink_patch_ids.extend(active_patch_ids)
                active_point_ids = []
                active_patch_ids = []

                break
            len_log = len(active_point_ids)

    return cloud

    # grow cluster

