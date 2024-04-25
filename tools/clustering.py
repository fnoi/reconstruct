import matplotlib.pyplot as plt
import numpy as np

from tools.local import neighborhood_search


def region_growing(cloud, config):
    """region growing algorithm for instance segmentation"""
    # actual annotation
    cloud['instance_pr'] = 0
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

        while True:

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

            # scatter plot potential cloud
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(potential_cloud['x'], potential_cloud['y'], potential_cloud['z'], s=1, c='grey', alpha=0.4)
            ax.scatter(cloud.loc[active_point_ids, 'x'], cloud.loc[active_point_ids, 'y'], cloud.loc[active_point_ids, 'z'], s=4, c='violet')
            ax.set_aspect('equal')
            plt.show()

            actual_neighbors = []
            patch_sn = seed_sn

            for active_point in active_point_ids:
                actual_neighbors.extend(neighborhood_search(
                    cloud=cloud, seed_id=active_point, config=config,
                    step='patch growing', patch_sn=patch_sn
                ))
            actual_neighbors = list(set(actual_neighbors))
            actual_neighbors = [_ for _ in actual_neighbors if _ not in active_point_ids]

            # scatter plot actual neighbors
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # 1. potential cloud
            ax.scatter(potential_cloud['x'], potential_cloud['y'], potential_cloud['z'], s=1, c='grey', alpha=0.4)
            # 2. active cloud
            ax.scatter(cloud.loc[active_point_ids, 'x'], cloud.loc[active_point_ids, 'y'], cloud.loc[active_point_ids, 'z'], s=4, c='violet')
            # 3. actual neighbors
            ax.scatter(cloud.loc[actual_neighbors, 'x'], cloud.loc[actual_neighbors, 'y'], cloud.loc[actual_neighbors, 'z'], s=4, c='green')
            ax.set_aspect('equal')
            plt.show()

            # for each of those neighbors, if they belong to a patch, check if patch can be added
            for neighbor in actual_neighbors:
                if neighbor in visited_neighbors:
                    continue
                else:
                    neighbor_patch = cloud.loc[neighbor, 'ransac_patch']
                    if neighbor_patch == 0:
                        # check if neighbor point can be added
                        a = 0
                    else:
                        # check if patch can be added
                        cluster_sn = seed_sn  ##
                        neighbor_patch_sn = cloud.loc[neighbor, ['snx', 'sny', 'snz']].values
                        # check if patch can be added
                        sn_dev = np.linalg.norm(cluster_sn - neighbor_patch_sn)

                        a = 0

        a = 0



        # grow cluster


