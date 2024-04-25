import matplotlib.pyplot as plt
import numpy as np

from tools.local import neighborhood_search, angular_deviation, supernormal_svd


def region_growing(cloud, config):
    """region growing algorithm for instance segmentation"""
    # actual annotation
    label_instance = 1
    cloud['instance_pr'] = 0  # unlabeled
    # source: unsegmented
    source_point_ids = cloud.index.tolist()
    source_patch_ids = cloud['ransac_patch'].unique().tolist()
    # active: now growing, assigned to current segment
    active_point_ids = []
    active_patch_ids = []
    # sink: already segmented
    sink_point_ids = []
    sink_patch_ids = []

    while True:
        if len(source_point_ids) <= config.region_growing.leftover_thresh:
            break
        # find seed point
        source_cloud = cloud.loc[source_point_ids]
        # sort source cloud by confidence
        source_cloud.sort_values(by='confidence', ascending=False, inplace=True)
        # seed point
        for i, row in source_cloud.iterrows():
            patch = row['ransac_patch']
            if patch == 0:
                continue
            else:
                seed_point_id = i
                break

        if seed_point_id == 1546:
            break
        seed_point_id = cloud.index[
            (cloud['x'] == source_cloud.loc[seed_point_id, 'x']) &
            (cloud['y'] == source_cloud.loc[seed_point_id, 'y']) &
            (cloud['z'] == source_cloud.loc[seed_point_id, 'z'])
        ].tolist()[0]

        seed_sn = cloud.loc[seed_point_id, ['snx', 'sny', 'snz']].values
        seed_rn = cloud.loc[seed_point_id, ['rnx', 'rny', 'rnz']].values

        # seed patch
        seed_patch_id = cloud.loc[seed_point_id, 'ransac_patch']
        seed_patch_point_ids = cloud.index[cloud['ransac_patch'] == seed_patch_id].tolist()

        # inventory
        active_point_ids.extend(seed_patch_point_ids)
        source_point_ids = [_ for _ in source_point_ids if _ not in seed_patch_point_ids]

        active_patch_ids.append(cloud.loc[seed_point_id, 'ransac_patch'])
        source_patch_ids.remove(cloud.loc[seed_point_id, 'ransac_patch'])

        cluster_blacklist = []
        growth_iter = 0
        len_log = 0

        while True:
            growth_iter += 1
            candidate_point_ids = [_ for _ in active_point_ids
                                   if _ not in sink_point_ids
                                   and _ not in cluster_blacklist]
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
            potential_cloud = cloud.loc[potential_neighbors]

            # # scatter plot potential cloud
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(potential_cloud['x'], potential_cloud['y'], potential_cloud['z'], s=1, c='grey', alpha=0.4)
            # ax.scatter(cloud.loc[active_point_ids, 'x'], cloud.loc[active_point_ids, 'y'], cloud.loc[active_point_ids, 'z'], s=4, c='violet')
            # ax.set_aspect('equal')
            # plt.show()

            actual_neighbors = []
            smart_choices = True
            if smart_choices and growth_iter > 1:
                # calculate supernormal based on cluster points
                cluster_normals = cloud.loc[active_point_ids, ['nx', 'ny', 'nz']].to_numpy()
                cluster_sn = supernormal_svd(cluster_normals)
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
                ransac_patch_sizes = cloud.loc[active_point_ids, 'ransac_patch'].value_counts()
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

            for active_point in active_point_ids:
                actual_neighbors.extend(neighborhood_search(
                    cloud=cloud, seed_id=active_point, config=config,
                    step='patch growing', patch_sn=cluster_sn
                ))
            actual_neighbors = list(set(actual_neighbors))
            actual_neighbors = [_ for _ in actual_neighbors if _ not in active_point_ids]

            # # scatter plot actual neighbors
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # # 1. potential cloud
            # ax.scatter(potential_cloud['x'], potential_cloud['y'], potential_cloud['z'], s=1, c='grey', alpha=0.4)
            # # 2. active cloud
            # ax.scatter(cloud.loc[active_point_ids, 'x'], cloud.loc[active_point_ids, 'y'], cloud.loc[active_point_ids, 'z'], s=4, c='violet')
            # # 3. actual neighbors
            # ax.scatter(cloud.loc[actual_neighbors, 'x'], cloud.loc[actual_neighbors, 'y'], cloud.loc[actual_neighbors, 'z'], s=4, c='green')
            # ax.set_aspect('equal')
            # plt.show()

            visited_neighbors = []
            # for each of those neighbors, if they belong to a patch, check if patch can be added
            for neighbor in actual_neighbors:
                if (neighbor in visited_neighbors
                        or cloud.loc[neighbor, 'ransac_patch'] in active_patch_ids
                        or neighbor in sink_point_ids
                ):
                    continue
                else:
                    neighbor_patch = cloud.loc[neighbor, 'ransac_patch']
                    if neighbor_patch == 0:
                        # check if neighbor point can be added
                        neighbor_sn = cloud.loc[neighbor, ['snx', 'sny', 'snz']].values
                        deviation_sn = angular_deviation(cluster_sn, neighbor_sn) % 180
                        deviation_sn = min(deviation_sn, 180 - deviation_sn)
                        if deviation_sn < config.region_growing.supernormal_point_angle_deviation:
                            active_point_ids.append(neighbor)
                            source_point_ids.remove(neighbor)
                    else:
                        # check if patch can be added
                        neighbor_patch_sn = cloud.loc[neighbor, ['snx', 'sny', 'snz']].values
                        neighbor_patch_rn = cloud.loc[neighbor, ['rnx', 'rny', 'rnz']].values

                        deviation_sn = angular_deviation(cluster_sn, neighbor_patch_sn) % 180
                        deviation_sn = min(deviation_sn, 180 - deviation_sn)
                        deviation_rn = angular_deviation(cluster_rn, neighbor_patch_rn) % 90
                        deviation_rn = min(deviation_rn, 90 - deviation_rn)

                        debug_plot = False
                        if debug_plot:
                            deviation_check = deviation_sn < config.region_growing.supernormal_patch_angle_deviation and \
                                    deviation_rn < config.region_growing.ransacnormal_patch_angle_deviation

                            # plot current cluster, neighbor patch and sn and rn each
                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection='3d')
                            # 0. potential cloud
                            ax.scatter(potential_cloud['x'],
                                       potential_cloud['y'],
                                       potential_cloud['z'],
                                       s=.2, c='grey', alpha=0.4)
                            # 1. cluster points
                            ax.scatter(cloud.loc[active_point_ids, 'x'],
                                       cloud.loc[active_point_ids, 'y'],
                                       cloud.loc[active_point_ids, 'z'],
                                       s=1, c='green')
                            # 2. neighbor patch points
                            ax.scatter(cloud.loc[cloud['ransac_patch'] == neighbor_patch, 'x'],
                                       cloud.loc[cloud['ransac_patch'] == neighbor_patch, 'y'],
                                       cloud.loc[cloud['ransac_patch'] == neighbor_patch, 'z'],
                                       s=1, c='blue')
                            x_move = 0.5
                            # 3. cluster supernormal
                            ax.quiver(cloud.loc[active_point_ids[0], 'x'] + x_move,
                                      cloud.loc[active_point_ids[0], 'y'],
                                      cloud.loc[active_point_ids[0], 'z'],
                                      cluster_sn[0] * .3,
                                      cluster_sn[1] * .3,
                                      cluster_sn[2] * .3,
                                      color='red')
                            # 4. neighbor patch supernormal from first point of neighbor patch
                            ax.quiver(cloud.loc[cloud['ransac_patch'] == neighbor_patch, 'x'].iloc[0] + x_move,
                                      cloud.loc[cloud['ransac_patch'] == neighbor_patch, 'y'].iloc[0],
                                      cloud.loc[cloud['ransac_patch'] == neighbor_patch, 'z'].iloc[0],
                                      neighbor_patch_sn[0] * .3,
                                      neighbor_patch_sn[1] * .3,
                                      neighbor_patch_sn[2] * .3,
                                      color='orange')
                            # 5. cluster ransac normal
                            ax.quiver(cloud.loc[active_point_ids[0], 'x'] + x_move,
                                      cloud.loc[active_point_ids[0], 'y'],
                                      cloud.loc[active_point_ids[0], 'z'],
                                      cluster_rn[0] * .3,
                                      cluster_rn[1] * .3,
                                      cluster_rn[2] * .3,
                                      color='purple')
                            # 6. neighbor patch ransac normal from first point of neighbor patch
                            ax.quiver(cloud.loc[cloud['ransac_patch'] == neighbor_patch, 'x'].iloc[0] + x_move,
                                      cloud.loc[cloud['ransac_patch'] == neighbor_patch, 'y'].iloc[0],
                                      cloud.loc[cloud['ransac_patch'] == neighbor_patch, 'z'].iloc[0],
                                      neighbor_patch_rn[0] * .3,
                                      neighbor_patch_rn[1] * .3,
                                      neighbor_patch_rn[2] * .3,
                                      color='yellow')

                            ax.set_aspect('equal')
                            # title
                            plt.title(f'{active_patch_ids} - {neighbor_patch} ::  d_sn: {deviation_sn:.1f};   d_rn: {deviation_rn:.1f}:\n{deviation_check}')
                            # legend below the figure
                            # plt.legend(['potential cloud', 'cluster points', 'neighbor patch points',
                            #             'cluster supernormal', 'neighbor patch supernormal',
                            #             'cluster ransac normal', 'neighbor patch ransac normal'],
                            #            loc='upper center', bbox_to_anchor=(0.5, -0.05),
                            #            fancybox=True, shadow=True, ncol=1)

                            plt.tight_layout()
                            plt.show()

                        if deviation_sn < config.region_growing.supernormal_patch_angle_deviation and \
                                deviation_rn < config.region_growing.ransacnormal_patch_angle_deviation:
                            active_patch_ids.append(neighbor_patch)
                            active_point_ids.extend(cloud.index[cloud['ransac_patch'] == neighbor_patch].tolist())
                            visited_neighbors.extend(cloud.index[cloud['ransac_patch'] == neighbor_patch].tolist())
                            if neighbor_patch not in source_patch_ids:
                                a = 0
                            source_patch_ids.remove(neighbor_patch)
                            source_point_ids = [
                                _ for _ in source_point_ids
                                if _ not in cloud.index[cloud['ransac_patch'] == neighbor_patch].tolist()
                            ]
                        else:
                            visited_neighbors.extend(cloud.index[cloud['ransac_patch'] == neighbor_patch].tolist())

            if len_log == len(active_point_ids):

                # scatter plot active cloud
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                # 1. full cloud
                ax.scatter(cloud['x'], cloud['y'], cloud['z'], s=.2, c='grey', alpha=0.4)
                # 2. active cloud
                ax.scatter(cloud.loc[active_point_ids, 'x'], cloud.loc[active_point_ids, 'y'], cloud.loc[active_point_ids, 'z'], s=1, c='violet')
                ax.set_aspect('equal')
                plt.show()


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


