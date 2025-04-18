import copy
import math
import random

import numpy as np
import open3d as o3d
import pandas as pd
from ifcopenshell.api.cost import add_cost_item
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.linalg import svd
from sklearn.cluster import DBSCAN

from tools.utils import plot_patch


def supernormal_svd_s1(normals, full_return=False):
    """
    Compute supernormal using SVD, returns None values if computation fails.

    Args:
        normals: Nx3 array of normal vectors
        full_return: If True, return additional SVD values
    """
    try:
        if np.isnan(normals).any():
            return (None, None, None, None) if full_return else None

        normals = consistency_flip(normals)

        if np.isnan(normals).any():
            return (None, None, None, None) if full_return else None

        U, S, Vt = svd(normals, full_matrices=False)

        # Check if we have less than 3 singular values
        if len(S) < 3:
            return (None, None, None, None) if full_return else None

        sig_1 = S[0]
        sig_2 = S[1]
        sig_3 = S[2]

        if full_return:
            return Vt[-1, :], sig_1, sig_2, sig_3
        else:
            return Vt[-1, :]

    except:
        return (None, None, None, None) if full_return else None


def consistency_flip(vectors):
    # Ensure input is a numpy array
    vectors = np.asarray(vectors)

    # Convert input to 2D array if it's a single vector
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    # Check if vectors is empty
    if vectors.size == 0:
        return vectors

    # normalize vectors
    try:
        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    except TypeError:
        # If TypeError occurs, try converting to float
        vectors = vectors.astype(float)
        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    vectors_norm = np.divide(vectors, vector_norms, where=vector_norms != 0)

    # mean resultant vector
    vector_mean = np.mean(vectors_norm, axis=0)
    mean_norm = np.linalg.norm(vector_mean)
    if mean_norm == 0:
        print('zero vector mean')
        return vectors_norm
    vector_mean /= mean_norm

    # flip vectors that have opposite orientation
    dot_products = np.dot(vectors_norm, vector_mean)
    vectors_flipped = np.where(dot_products[:, np.newaxis] < 0, -vectors, vectors)

    # If input was a single vector, return a 1D array
    if vectors.shape[0] == 1:
        return vectors_flipped.flatten()

    return vectors_flipped


def supernormal_confidence(supernormal, normals, sig_1, sig_2, sig_3):
    """
    calculate confidence value of supernormal,
    confidence = md_sn / md_n
    md_sn: mean deviation angle between supernormal and normals (-90)
    md_n:  mean deviation angle between normals
    big sn, small n -> big confidence
    big sn, big n   -> small confidence
    small sn, big n -> very small confidence
    small sn, small n -> big confidence
    (is the idea, lets check)
    """
    supernormal /= np.linalg.norm(supernormal)

    # norms = np.linalg.norm(normals, axis=1)[:, None]
    # norms = norms + 1e-10  # avoid division by zero
    # normals /= norms
    #
    # # deviation between supernormal and normals (from being perpendicular) # TODO: revise confidence calculation and equation in paper
    # # normals /= np.linalg.norm(normals, axis=1)[:, None]
    # n_sn_90_dev = np.arccos(np.dot(supernormal, normals.T))
    # n_sn_90_dev = np.abs(np.rad2deg(n_sn_90_dev))
    # n_sn_90_dev = np.abs(n_sn_90_dev - 90)
    # n_sn_90_dev = np.clip(n_sn_90_dev, 0, 90)
    # n_sn_90_dev = n_sn_90_dev / 90
    # n_sn_90_dev = np.mean(n_sn_90_dev)
    # n_sn_90_dev = 1 - n_sn_90_dev
    #
    # # def from reference / mean normal
    # normals_flipped = consistency_flip(normals)
    # n_ref = np.mean(normals_flipped, axis=0)
    # n_ref /= np.linalg.norm(n_ref)
    #
    # n_dev = np.arccos(np.dot(n_ref, normals.T))
    # n_dev = np.abs(np.rad2deg(n_dev))
    # n_dev = np.clip(n_dev, 0, 90)
    # n_dev = np.mean(n_dev)
    #
    # c = math.sqrt(math.sqrt(len(normals)) * n_dev * n_sn_90_dev)

    # c = math.sqrt(len(normals)) * (sig_3/sig_1)

    c = math.sqrt(len(normals)) * (sig_2 / sig_1) * (1 - sig_3 / sig_2)

    # c = np.mean(n_sn_90_dev)
    #
    # c *= len(normals)

    return c


def angular_deviation(vector, reference):
    if vector is None or reference is None:
        raise ValueError("input vectors must be non-zero.")
    norm_vector = np.linalg.norm(vector)
    norm_reference = np.linalg.norm(reference)
    if norm_vector == 0 or norm_reference == 0:
        raise ValueError("input vectors must be non-zero.")

    vector_normalized = vector / norm_vector
    reference_normalized = reference / norm_reference

    vector_normalized = vector_normalized.flatten()
    reference_normalized = reference_normalized.flatten()

    # compute dot product and clamp it to the valid range for arccos
    dot_product = np.dot(vector_normalized, reference_normalized)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # compute the angle in radians and then convert to degrees
    angle = np.arccos(dot_product)
    return angle * 180 / np.pi


def neighborhood_calculations(cloud=None, cloud_tree=None, seed_id=None, config=None, plot_ind=None, plot_flag=False):
    neighbor_ids = neighborhood_search(cloud_tree=cloud_tree, seed_id=seed_id, config=config, cloud=cloud)
    # neighbor_ids = neighborhood_search(cloud, seed_id, config)
    # 3d scatter all points, color seed red, neighbors blue

    if plot_flag and seed_id == plot_ind:
        neighborhood_plot(cloud, seed_id, neighbor_ids, config)

    rnd_picks_abs = True
    if rnd_picks_abs:
        neighbor_cap = 100
        if len(neighbor_ids) > 100:
            neighbor_ids = random.sample(neighbor_ids, 100)

    sort_flag = False
    if cloud.loc[seed_id, 'confidence'] is not None and sort_flag:
        # sort neighborhood_ids by confidence
        neighbor_confidences = cloud.loc[neighbor_ids, 'confidence']
        neighbor_ids = neighbor_ids[np.argsort(neighbor_confidences)]
        # choose the worst 80% of the neighborhood
        relative_neighborhood_size = 0.6
        start_index = int(len(neighbor_ids) * (1 - relative_neighborhood_size))
        neighbor_ids = neighbor_ids[start_index:]

    if config.supernormal.input == 'ransac':
        # neighbor_normals = cloud.loc[neighbor_ids][['rnx', 'rny', 'rnz']].values
        # use 'id' column
        neighbor_normals = cloud[cloud['id'].isin(neighbor_ids)][['rnx', 'rny', 'rnz']].values
        # remove zero vectors
        neighbor_normals = neighbor_normals[np.linalg.norm(neighbor_normals, axis=1) > 0]
        if len(neighbor_normals) < 10:
            print('found less than 10 neighbors with ransac normals, using normals instead')
            neighbor_normals = cloud[cloud['id'].isin(neighbor_ids)][['nx', 'ny', 'nz']].values

    else:
        # neighbor_normals = cloud.iloc[neighbor_ids][['nx', 'ny', 'nz']].values
        # use 'id' column
        neighbor_normals = cloud[cloud['id'].isin(neighbor_ids)][['nx', 'ny', 'nz']].values


    seed_supernormal, sig_1, sig_2, sig_3 = supernormal_svd_s1(neighbor_normals, full_return=True)
    seed_supernormal /= np.linalg.norm(seed_supernormal)

    cloud.loc[cloud['id'] == seed_id, 'snx'] = seed_supernormal[0]
    cloud.loc[cloud['id'] == seed_id, 'sny'] = seed_supernormal[1]
    cloud.loc[cloud['id'] == seed_id, 'snz'] = seed_supernormal[2]
    cloud.loc[cloud['id'] == seed_id, 'confidence'] = supernormal_confidence(seed_supernormal, neighbor_normals, sig_1, sig_2, sig_3)
    # cloud.loc[seed_id, 'snx'] = seed_supernormal[0]
    # cloud.loc[seed_id, 'sny'] = seed_supernormal[1]
    # cloud.loc[seed_id, 'snz'] = seed_supernormal[2]
    # cloud.loc[seed_id, 'confidence'] = supernormal_confidence(seed_supernormal, neighbor_normals)

    return cloud


def calculate_supernormals_rev(cloud=None, cloud_tree=None, config=None):
    cloud['snx'] = None
    cloud['sny'] = None
    cloud['snz'] = None
    cloud['confidence'] = None

    plot_flag = True
    plot_ind = 'a'
    if plot_flag:
        print(f'point {plot_ind} will be used for neighborhood plot')

    point_ids = np.arange(len(cloud))

    if config.local_neighborhood.shape in ['cube', 'sphere', 'ellipsoid']:  # unoriented neighborhoods
        for seed_id in tqdm(point_ids, desc="computing supernormals, one step", total=len(point_ids)):
                cloud = neighborhood_calculations(cloud=cloud, cloud_tree=cloud_tree, seed_id=seed_id, config=config,
                                                  plot_ind=plot_ind, plot_flag=plot_flag)
        # no second step of computation needed

    elif config.local_features.neighbor_shape in ['oriented_ellipsoid', 'oriented_cylinder', 'oriented_cuboid']:
        for seed_id in tqdm(point_ids, desc="computing supernormals (1/2)", total=len(point_ids)):

            real_config = copy.copy(config)  # save config
            config.local_features.neighbor_shape = "sphere"  # override for precomputation
            cloud = neighborhood_calculations(cloud=cloud, seed_id=seed_id, config=config,
                                              plot_ind=plot_ind, plot_flag=plot_flag)
            config = real_config  # reset config

        for seed_id in tqdm(point_ids, desc="computing supernormals (2/2)", total=len(point_ids)):
            # oriented neighborhoods require supernormals as input
            cloud = neighborhood_calculations(cloud=cloud, seed_id=seed_id, config=config,
                                              plot_ind=plot_ind, plot_flag=plot_flag)

    else:
        raise ValueError(f'neighborhood shape "{config.local_features.neighbor_shape}" not implemented')

    return cloud


def ransac_patches(cloud, config):
    print('ransac patching')
    cloud['rnx'] = 0.0
    cloud['rny'] = 0.0
    cloud['rnz'] = 0.0
    cloud['ransac_patch'] = 0
    mask_remaining = np.ones(len(cloud), dtype=bool)
    progress = tqdm(total=len(cloud))
    blanks = config.planar_patch.ransac_blank_ok

    label_id = 0  # label 0 indicates unclustered
    o3d_cloud = o3d.geometry.PointCloud()

    while np.sum(mask_remaining) > 0:
        current_points = cloud.loc[mask_remaining, ['x', 'y', 'z']].values
        # convert to datatype float
        current_points = current_points.astype(np.float32)
        o3d_cloud.points = o3d.utility.Vector3dVector(current_points)

        # ransac plane fitting
        ransac_plane, ransac_inliers = o3d_cloud.segment_plane(
            distance_threshold=config.planar_patch.ransac_distance_threshold,
            ransac_n=config.planar_patch.ransac_ransac_n,
            num_iterations=config.planar_patch.ransac_num_iterations,

        )
        ransac_normal = ransac_plane[0:3]
        inliers_global_idx = np.where(mask_remaining)[0][ransac_inliers]

        # dbscan clustering of inliers
        dbscan_clustering = DBSCAN(
            eps=config.planar_patch.dbscan_eps,
            min_samples=config.planar_patch.dbscan_min_samples
        ).fit(current_points[ransac_inliers])

        clusters = np.unique(dbscan_clustering.labels_)

        if len(clusters) == 1 and clusters[0] == -1:
            blanks -= 1
            if blanks == 0:
                print(f'over, {label_id} patches found')
                break
            continue

        # remove -1 from clusters
        clusters = clusters[clusters != -1]
        # find biggest cluster
        cluster_sizes = np.array([np.sum(dbscan_clustering.labels_ == cluster) for cluster in clusters])
        biggest_cluster = clusters[np.argmax(cluster_sizes)]

        label_id += 1
        cluster_idx = np.where(dbscan_clustering.labels_ == biggest_cluster)[0]
        external_cluster_idx = inliers_global_idx[cluster_idx]
        cloud.loc[external_cluster_idx, 'ransac_patch'] = label_id
        cloud.loc[external_cluster_idx, ['rnx', 'rny', 'rnz']] = ransac_normal
        mask_remaining[external_cluster_idx] = False
        progress.update(len(external_cluster_idx))

    print(f'remaining points {np.sum(mask_remaining)} of {len(mask_remaining)}')
    progress.close()

    debug_plot = True
    if debug_plot:
        # plot histogram for ransac_patch
        plt.hist(cloud['ransac_patch'], bins=label_id)
        plt.show()

        # create plot with two subplots
        ax_lims_x = (np.min(cloud['x']), np.max(cloud['x']))
        ax_lims_y = (np.min(cloud['y']), np.max(cloud['y']))
        ax_lims_z = (np.min(cloud['z']), np.max(cloud['z']))

        plt.figure()
        mask_done = np.invert(mask_remaining)
        cloud_clustered = cloud.loc[mask_done]
        ax1 = plt.subplot(121, projection='3d')
        ax1.scatter(cloud_clustered['x'], cloud_clustered['y'], cloud_clustered['z'],
                    c=plt.cm.tab10(np.mod(cloud_clustered['ransac_patch'], 10)),
                    s=.5)
        ax1.set_xlim(ax_lims_x)
        ax1.set_ylim(ax_lims_y)
        ax1.set_zlim(ax_lims_z)

        cloud_unclustered = cloud.loc[mask_remaining]
        ax2 = plt.subplot(122, projection='3d')
        ax2.scatter(cloud_unclustered['x'], cloud_unclustered['y'], cloud_unclustered['z'], c='grey', s=.5)
        ax1.set_xlim(ax_lims_x)
        ax1.set_ylim(ax_lims_y)
        ax1.set_zlim(ax_lims_z)
        # equal aspect ratio
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        # set tight
        plt.tight_layout()
        plt.show()

    return cloud


def cluster_extent(cloud, patch_ids):
    """
    calculate the extent of a cluster
    :param cloud: point cloud
    :param patch_ids: list of ransac_patch_ids
    :return: extent of cluster
    """
    cluster_points = cloud.loc[cloud['ransac_patch'].isin(patch_ids)]
    cluster_lims = [
        np.min(cluster_points['x']),
        np.max(cluster_points['x']),
        np.min(cluster_points['y']),
        np.max(cluster_points['y']),
        np.min(cluster_points['z']),
        np.max(cluster_points['z'])
    ]
    return cluster_lims


def grow_stage_1(cloud, config):
    cloud['grown_patch'] = 0
    patch_label = 0
    cloud_local = cloud.copy()
    # mask points that have ransac patch 0
    cloud_local = cloud_local[cloud_local['ransac_patch'] != 0]
    patch_source = np.unique(cloud_local['ransac_patch'])
    patch_sink = []
    patch_active = []

    patch_dict = {}
    for ransac_patch in patch_source:
        center_point = np.mean(
            cloud_local.loc[cloud_local['ransac_patch'] == ransac_patch, ['x', 'y', 'z']].values,
            axis=0
        )
        # find the row in the patch with max confidence
        confident_point = cloud_local.loc[cloud_local['ransac_patch'] == ransac_patch].idxmax()['confidence']

        patch_dict[ransac_patch] = {    # this would be a good place to reduce sn influence by voting
            'cx': center_point[0],
            'cy': center_point[1],
            'cz': center_point[2],
            'snx': cloud_local.loc[confident_point]['snx'],
            'sny': cloud_local.loc[confident_point]['sny'],
            'snz': cloud_local.loc[confident_point]['snz'],
            'rnx': cloud_local.loc[confident_point]['rnx'],
            'rny': cloud_local.loc[confident_point]['rny'],
            'rnz': cloud_local.loc[confident_point]['rnz'],
            'max_confidence': cloud_local.loc[confident_point]['confidence']
        }
    patch_data = pd.DataFrame.from_dict(patch_dict, orient='index')

    while True:
        # find max confidence patch for those patches in source
        patch_data_masked = patch_data[patch_data.index.isin(patch_source)]
        max_confidence_patch = patch_data_masked.idxmax()['max_confidence']
        patch_active.append(max_confidence_patch)
        max_confidence_patch_data = patch_data.loc[max_confidence_patch]

        while True:
            # find closest patch
            patch_data_masked = patch_data[patch_data.index.isin(patch_source)]
            patch_data_masked['distance'] = np.sqrt(
                (patch_data_masked['cx'] - max_confidence_patch_data['cx']) ** 2 +
                (patch_data_masked['cy'] - max_confidence_patch_data['cy']) ** 2 +
                (patch_data_masked['cz'] - max_confidence_patch_data['cz']) ** 2
            )




    a = 0

    while True:
        if len(patch_source) == 0:
            plot_patch_v3(cloud, num_colors=patch_label)
            break

        seed = cloud_local.idxmax()['confidence']
        seed_patch = cloud_local.loc[seed, 'ransac_patch']

        while True:
            print(f'patch {seed_patch} growing')




    a = 0


def patch_growing(cloud, config):
    cloud['grown_patch'] = 0
    label_id = 0
    mask_available = np.ones(len(cloud), dtype=bool)
    patches_clustered = []
    clustered_tracker = []

    while True:

        # count trues in mask_available
        if np.sum(mask_available) <= config.region_growing.leftover_thresh * len(cloud):
            plot_patch_v3(cloud, num_colors=label_id)
            break

        masked_cloud = cloud.loc[mask_available]
        seed = masked_cloud.idxmax()['confidence']
        seed_data = masked_cloud.loc[seed]

        # find seed in full unmasked cloud
        seed = cloud.index[
            (cloud['x'] == seed_data['x']) &
            (cloud['y'] == seed_data['y']) &
            (cloud['z'] == seed_data['z'])
            ][0]

        seed_patch = cloud.loc[seed, 'ransac_patch']
        cluster_sn = cloud.loc[seed, ['snx', 'sny', 'snz']].values
        cluster_rn = cloud.loc[seed, ['rnx', 'rny', 'rnz']].values  # unused?
        # int list of point ids in cluster
        cluster_points = cloud.loc[cloud['ransac_patch'] == seed_patch].index
        cluster_points = cluster_points.tolist()
        cluster_patches = [int(cloud.loc[seed, 'ransac_patch'])]
        cluster_patches_out = []

        while True:
            # identify neighboring clusters through neighboring points (as before)
            neighbor_points = []

            cluster_lims = [np.min(cloud.loc[cluster_points, 'x']),
                            np.max(cloud.loc[cluster_points, 'x']),
                            np.min(cloud.loc[cluster_points, 'y']),
                            np.max(cloud.loc[cluster_points, 'y']),
                            np.min(cloud.loc[cluster_points, 'z']),
                            np.max(cloud.loc[cluster_points, 'z'])]

            # TODO: neighborhood search is bottleneck...
            #  mask cloud for speed? (already checked / pre-filter box)
            neighbor_ids = neighborhood_search(cloud, seed_id=None, config=config,
                                               step='bbox_mask', cluster_lims=cluster_lims)
            # masked_cloud = cloud.loc[mask_ids]
            # for cluster_point in cluster_points:
            #     neighbor_points.extend(neighborhood_search(masked_cloud, cluster_point, config, step='patch growing'))
            # neighbor_points = np.unique(neighbor_points)
            neighbor_patches = np.unique(cloud.loc[neighbor_ids, 'ransac_patch'])
            neighbor_patches = [x for x in neighbor_patches if
                                x != 0 and
                                x not in cluster_patches and
                                x not in cluster_patches_out and
                                x not in patches_clustered]
            if len(neighbor_patches) == 0:
                # patch growth stopped.
                label_id += 1
                cloud.loc[cluster_points, 'grown_patch'] = label_id
                mask_available[cluster_points] = False

                print(f'patch grown: {label_id}, size: {len(cluster_points)}')
                break

            # check if neighboring patches are cool
            # criteria: 1. angle between sn and sn (should be 0) 2. angle between sn (0) and rn (1) should be 90
            for neighbor_patch in neighbor_patches:
                patch_center_point = np.mean(
                    cloud.loc[cloud['ransac_patch'] == neighbor_patch, ['x', 'y', 'z']].values,
                    axis=0
                )
                patch_ids = cloud.loc[cloud['ransac_patch'] == neighbor_patch].index

                # patch supernormal
                neighbor_patch_sn = np.mean(
                    cloud.loc[cloud['ransac_patch'] == neighbor_patch, ['snx', 'sny', 'snz']].values,
                    axis=0
                )
                neighbor_patch_sn /= np.linalg.norm(neighbor_patch_sn)

                sn_deviation = angular_deviation(cluster_sn, neighbor_patch_sn)
                sn_deviation = min(
                    min(abs(sn_deviation - 0), abs(sn_deviation - 360)),
                    abs(sn_deviation - 180)
                )

                # patch ransac normal
                neighbor_patch_rn = np.mean(
                    cloud.loc[cloud['ransac_patch'] == neighbor_patch, ['rnx', 'rny', 'rnz']].values,
                    axis=0
                )
                neighbor_patch_rn /= np.linalg.norm(neighbor_patch_rn)

                rn_deviation = angular_deviation(cluster_sn, neighbor_patch_rn) - 90

                # grow_plot_v2(cloud, cluster_points, patch_ids,
                #              cluster_sn, neighbor_patch_sn,
                #              cluster_rn, neighbor_patch_rn,
                #              seed, patch_center_point)

                report_flag = False
                if report_flag:
                    print('sn deviation: ', sn_deviation, 'rn deviation: ', rn_deviation)

                if sn_deviation <= config.region_growing.supernormal_patch_angle_deviation and \
                        rn_deviation <= config.region_growing.ransacnormal_patch_angle_deviation:
                    # add neighbor patch to cluster
                    cluster_points.extend(cloud.loc[cloud['ransac_patch'] == neighbor_patch].index)
                    cluster_patches.append(neighbor_patch)
                    patches_clustered.append(neighbor_patch)

                    # growth_plot(cloud, seed, cluster_points, [], [], [], [])
                    if report_flag:
                        print('cool')
                else:
                    cluster_patches_out.append(neighbor_patch)
                    if report_flag:
                        print('not cool')

                if report_flag:
                    print('cluster size: ', len(cluster_points))

    return cloud


def grow_plot_v2(cloud, cluster_points, patch_ids,
                 cluster_sn, neighbor_patch_sn,
                 cluster_rn, neighbor_patch_rn,
                 seed, patch_center_point):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cloud.loc[cluster_points, 'x'], cloud.loc[cluster_points, 'y'], cloud.loc[cluster_points, 'z'], s=0.3, c='grey')
    ax.scatter(cloud.loc[patch_ids, 'x'], cloud.loc[patch_ids, 'y'], cloud.loc[patch_ids, 'z'], s=0.6, c='blue')
    ax.scatter(cloud.loc[seed, 'x'], cloud.loc[seed, 'y'], cloud.loc[seed, 'z'], s=10, c='orange')
    ax.scatter(patch_center_point[0], patch_center_point[1], patch_center_point[2], s=10, c='green')
    ax.plot([cloud.loc[seed, 'x'], cloud.loc[seed, 'x'] + cluster_sn[0]],
            [cloud.loc[seed, 'y'], cloud.loc[seed, 'y'] + cluster_sn[1]],
            [cloud.loc[seed, 'z'], cloud.loc[seed, 'z'] + cluster_sn[2]],
            c='red', linewidth=2)
    ax.plot([cloud.loc[seed, 'x'], cloud.loc[seed, 'x'] + cluster_rn[0]],
            [cloud.loc[seed, 'y'], cloud.loc[seed, 'y'] + cluster_rn[1]],
            [cloud.loc[seed, 'z'], cloud.loc[seed, 'z'] + cluster_rn[2]],
            c='green', linewidth=2)
    ax.plot([patch_center_point[0], patch_center_point[0] + neighbor_patch_sn[0]],
            [patch_center_point[1], patch_center_point[1] + neighbor_patch_sn[1]],
            [patch_center_point[2], patch_center_point[2] + neighbor_patch_sn[2]],
            c='orange', linewidth=2)
    ax.plot([patch_center_point[0], patch_center_point[0] + neighbor_patch_rn[0]],
            [patch_center_point[1], patch_center_point[1] + neighbor_patch_rn[1]],
            [patch_center_point[2], patch_center_point[2] + neighbor_patch_rn[2]],
            c='blue', linewidth=2)
    ax.set_aspect('equal')

    # legend
    fig.legend(['seed supernormal',
                'seed ransac normal',
                'neighbor patch supernormal',
                'neighbor patch ransac normal',
                'seed',
                'neighbor patch center'],
               loc='center left', bbox_to_anchor=(1.0, 0.0))

    plt.show()


def plot_patch_v3(cloud, num_colors=None):
    plt.figure()

    if num_colors is None:
        n_colors = 10
    else:
        n_colors = num_colors
    cmap = plt.cm.get_cmap('gist_rainbow', n_colors)
    colors = cmap(np.arange(n_colors))
    colors[0] = np.array([0.8, 0.8, 0.8, 1.0])
    cmap = mcolors.ListedColormap(colors)

    norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n_colors + 0.5, 1), ncolors=n_colors)

    ax = plt.subplot(111, projection='3d')
    ax.scatter(cloud['x'], cloud['y'], cloud['z'], s=0.3, c=cloud['grown_patch'],
               cmap=plt.cm.tab20, vmin=0, vmax=10)

    scatter = ax.scatter(cloud['x'], cloud['y'], cloud['z'], s=0.3, c=cloud['grown_patch'], cmap=cmap,
                         norm=norm)

    # Add a colorbar to show the mapping from grown_patch values to colors
    cbar = plt.colorbar(scatter, ticks=np.arange(n_colors))
    cbar.set_label('Grown Patch Value')

    plt.show()


def neighborhood_search(seed_id, config, cloud=None, cloud_tree=None, step=None, cluster_lims=None, patch_sn=None, context=None):

    radius = None

    if step == 'bbox_mask':
        shape = "cube"
    elif step == 'patch growing':
        shape = config.region_growing.neighborhood_shape
        seed_data = cloud.loc[cloud['id'] == seed_id]
        radius = config.region_growing.neighborhood_radius
        # seed_data = cloud.iloc[seed_id]
    else:
        shape = config.local_neighborhood.shape
        # if cloud has 'id' column:
        if 'id' in cloud.columns:
            seed_data = cloud.loc[cloud['id'] == seed_id]
        else:
            seed_data = cloud.iloc[seed_id]

    if shape == "sphere":
        neighbor_ids = neighbors_sphere(cloud_tree, seed_data, config, context, radius)
    elif shape == "cylinder":
        neighbor_ids = neighbors_oriented_cylinder(cloud, seed_id, config)
    elif shape == "oriented_cuboid":
        neighbor_ids = neighbors_oriented_cuboid(cloud, seed_id, config)
    elif shape == "ellipsoid":
        neighbor_ids = neighbors_ellipsoid(cloud, seed_id, config)
    elif shape == "cube":
        neighbor_ids = neighbors_aabb_cube(cloud, seed_id, config, step, cluster_lims)
    elif shape == "oriented_ellipsoid":
        neighbor_ids = neighbors_oriented_ellipsoid(cloud, seed_id, config, step, patch_sn, seed_data)
    elif shape == "oriented_octahedron":
        neighbor_ids = neighbors_oriented_octahedron(cloud, seed_id, config)  # TODO: add this fct
    else:
        raise ValueError(f'neighborhood shape "{config.local_features.neighbor_shape}" not implemented')

    return neighbor_ids


def growth_plot(cloud, seed_id, cluster_active, cluster_passive, add_candidates, patch_additions, angle_checked):
    # create mask for cloud to exclude any id in clustered_inactive, add_candidates, patch_additions
    mask = np.ones(len(cloud), dtype=bool)

    mask[cluster_active] = False
    mask[add_candidates] = False
    mask[patch_additions] = False
    mask[cluster_passive] = False
    mask[angle_checked] = False
    mask[seed_id] = False

    # remove patch additions from cluster_active
    cluster_active = [x for x in cluster_active if x not in patch_additions]

    fig = plt.figure()

    subplots = [221, 222, 223, 224]
    perspectives = [(0, 0), (90, 90), (45, 225), (45, 315)]
    _ax = [fig.add_subplot(subplot, projection='3d') for subplot in subplots]

    for ax_ in _ax:
        ax_.scatter(cloud.loc[mask, 'x'],
                    cloud.loc[mask, 'y'],
                    cloud.loc[mask, 'z'],
                    s=0.2, c='grey')
        ax_.scatter(cloud.loc[cluster_passive, 'x'],
                    cloud.loc[cluster_passive, 'y'],
                    cloud.loc[cluster_passive, 'z'],
                    s=0.6, c='grey')
        ax_.scatter(cloud.loc[cluster_active, 'x'],
                    cloud.loc[cluster_active, 'y'],
                    cloud.loc[cluster_active, 'z'],
                    s=0.6, c='blue')
        ax_.scatter(cloud.loc[patch_additions, 'x'],
                    cloud.loc[patch_additions, 'y'],
                    cloud.loc[patch_additions, 'z'],
                    s=0.6, c='green')
        ax_.scatter(cloud.loc[add_candidates, 'x'],
                    cloud.loc[add_candidates, 'y'],
                    cloud.loc[add_candidates, 'z'],
                    s=0.6, c='yellow')
        ax_.scatter(cloud.loc[angle_checked, 'x'],
                    cloud.loc[angle_checked, 'y'],
                    cloud.loc[angle_checked, 'z'],
                    s=0.6, c='red')
        ax_.scatter(cloud.loc[seed_id, 'x'],
                    cloud.loc[seed_id, 'y'],
                    cloud.loc[seed_id, 'z'],
                    s=10, c='orange')
        ax_.view_init(elev=perspectives[_ax.index(ax_)][0], azim=perspectives[_ax.index(ax_)][1])
        ax_.set_aspect('equal')
        ax_.set_xticks([])
        ax_.set_yticks([])
        ax_.set_zticks([])

    fig.legend(['unclustered', 'cluster_passive', 'cluster_active', 'patch_additions', 'add_candidates', 'angle_checked', 'seed'],
               loc='center left', bbox_to_anchor=(0.4, 0.5))
    fig.tight_layout()
    fig.set_size_inches(7, 7)
    plt.show()


def neighborhood_plot(cloud, seed_id=None, neighbors=None, config=None, cage_override=None, confidence=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cloud['x'], cloud['y'], cloud['z'], s=0.3, c='grey')

    if neighbors is not None:
        ax.scatter(cloud.loc[neighbors, 'x'], cloud.loc[neighbors, 'y'], cloud.loc[neighbors, 'z'], s=1, c='r')

    if confidence:
        ax.scatter(cloud['x'], cloud['y'], cloud['z'], s=0.3, c=cloud['confidence'], cmap='viridis')

    cage_color = 'b'
    cage_width = 0.5
    if cage_override is not None:
        config = None

    if config is not None:
        case = config.local_features.neighbor_shape
        if case == "cube":
            # cornerpoints from seed and supernormal_cube_dist as edge length of cube
            center = cloud.loc[seed_id, ['x', 'y', 'z']].values
            edge_length = config.local_features.supernormal_cube_dist
            p0 = center + np.array([-edge_length, -edge_length, -edge_length])
            p1 = center + np.array([-edge_length, -edge_length, edge_length])
            p2 = center + np.array([-edge_length, edge_length, -edge_length])
            p3 = center + np.array([-edge_length, edge_length, edge_length])
            p4 = center + np.array([edge_length, -edge_length, -edge_length])
            p5 = center + np.array([edge_length, -edge_length, edge_length])
            p6 = center + np.array([edge_length, edge_length, -edge_length])
            p7 = center + np.array([edge_length, edge_length, edge_length])
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], c=cage_color, linewidth=cage_width)
            ax.plot([p1[0], p3[0]], [p1[1], p3[1]], [p1[2], p3[2]], c=cage_color, linewidth=cage_width)
            ax.plot([p3[0], p2[0]], [p3[1], p2[1]], [p3[2], p2[2]], c=cage_color, linewidth=cage_width)
            ax.plot([p2[0], p0[0]], [p2[1], p0[1]], [p2[2], p0[2]], c=cage_color, linewidth=cage_width)
            ax.plot([p4[0], p5[0]], [p4[1], p5[1]], [p4[2], p5[2]], c=cage_color, linewidth=cage_width)
            ax.plot([p5[0], p7[0]], [p5[1], p7[1]], [p5[2], p7[2]], c=cage_color, linewidth=cage_width)
            ax.plot([p7[0], p6[0]], [p7[1], p6[1]], [p7[2], p6[2]], c=cage_color, linewidth=cage_width)
            ax.plot([p6[0], p4[0]], [p6[1], p4[1]], [p6[2], p4[2]], c=cage_color, linewidth=cage_width)
            ax.plot([p0[0], p4[0]], [p0[1], p4[1]], [p0[2], p4[2]], c=cage_color, linewidth=cage_width)
            ax.plot([p1[0], p5[0]], [p1[1], p5[1]], [p1[2], p5[2]], c=cage_color, linewidth=cage_width)
            ax.plot([p2[0], p6[0]], [p2[1], p6[1]], [p2[2], p6[2]], c=cage_color, linewidth=cage_width)
            ax.plot([p3[0], p7[0]], [p3[1], p7[1]], [p3[2], p7[2]], c=cage_color, linewidth=cage_width)
        if case == "sphere":
            # plot sphere around seed point
            n_segments = 14
            center = cloud.loc[seed_id, ['x', 'y', 'z']].values
            radius = config.local_features.supernormal_radius
            u = np.linspace(0, 2 * np.pi, n_segments)
            v = np.linspace(0, np.pi, n_segments)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
            # ax.plot_surface(x, y, z, color='b', alpha=0.1)
            # wireframe
            for i in range(n_segments):
                ax.plot(x[i, :], y[i, :], z[i, :], c=cage_color, linewidth=cage_width)
                ax.plot(x[:, i], y[:, i], z[:, i], c=cage_color, linewidth=cage_width)

        if case == "ellipsoid":
            # plot ellipsoid around seed point
            n_segments = 14
            center = cloud.loc[seed_id, ['x', 'y', 'z']].values
            a = config.local_features.supernormal_ellipsoid_a  # Semi-major axis (along x)
            b = config.local_features.supernormal_ellipsoid_bc  # Semi-minor axes (along y and z)
            u = np.linspace(0, 2 * np.pi, n_segments)
            v = np.linspace(0, np.pi, n_segments)
            x = a * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = b * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = b * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
            # ax.plot_surface(x, y, z, color='b', alpha=0.1)
            # wireframe
            for i in range(n_segments):
                ax.plot(x[i, :], y[i, :], z[i, :], c=cage_color, linewidth=cage_width)
                ax.plot(x[:, i], y[:, i], z[:, i], c=cage_color, linewidth=cage_width)

        if case == "oriented_ellipsoid":
            n_segments = 14
            center = cloud.loc[seed_id, ['x', 'y', 'z']].values
            a = config.local_features.supernormal_ellipsoid_a  # Semi-major axis (along x)
            b = config.local_features.supernormal_ellipsoid_bc  # Semi-minor axes (along y and z)
            u = np.linspace(0, 2 * np.pi, n_segments)
            v = np.linspace(0, np.pi, n_segments)
            x = a * np.outer(np.cos(u), np.sin(v))
            y = b * np.outer(np.sin(u), np.sin(v))
            z = b * np.outer(np.ones(np.size(u)), np.cos(v))

            direction = cloud.loc[seed_id, ['snx', 'sny', 'snz']].values

            rot_mat = find_orthonormal_basis(direction)

            ellipsoid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
            rotated_points = rot_mat @ ellipsoid_points

            x_rotated = rotated_points[0, :].reshape(x.shape) + center[0]
            y_rotated = rotated_points[1, :].reshape(y.shape) + center[1]
            z_rotated = rotated_points[2, :].reshape(z.shape) + center[2]

            for i in range(n_segments):
                ax.plot(x_rotated[i, :], y_rotated[i, :], z_rotated[i, :], c=cage_color, linewidth=cage_width)
                ax.plot(x_rotated[:, i], y_rotated[:, i], z_rotated[:, i], c=cage_color, linewidth=cage_width)

        else:
            print(
                f'\nno cage plot implemented for the neighborhood shape of {config.local_features.neighbor_shape}')

    ax.set_aspect('equal')

    if seed_id is not None:
        ax.scatter(cloud.loc[seed_id, 'x'], cloud.loc[seed_id, 'y'], cloud.loc[seed_id, 'z'], s=10, c='orange')

    plt.show()


def find_orthonormal_basis(direction):
    """
    find orthonormal basis from direction vector
    """
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    if np.allclose(direction, [0, 0, 1]) or np.allclose(direction, [0, 1, 0]):
        other = np.array([1, 0, 1])
    else:
        other = np.array([0, 0, 1])
    y_axis = np.cross(direction, other)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(direction, y_axis)

    # return np.column_stack((direction, y_axis, z_axis))
    return np.vstack((direction, y_axis, z_axis))


def neighbors_oriented_ellipsoid(cloud, seed_id, config, step=None, patch_sn=None, seed_data=None):
    """
    find neighbors of seed_id in ellipsoid shape
    """
    seed_coords = seed_data[['x', 'y', 'z']].values
    cloud_coords = cloud[['x', 'y', 'z']].values

    if step == 'patch growing':
        a = config.region_growing.neighborhood_radius_a
        b = config.region_growing.neighborhood_radius_bc

        orientation = patch_sn

    else:
        a = config.local_features.supernormal_ellipsoid_a
        b = config.local_features.supernormal_ellipsoid_bc  # Semi-minor axis along y and z

        orientation = seed_data[['snx', 'sny', 'snz']].values

    rot_mat = find_orthonormal_basis(orientation)

    rotated_cloud = (cloud_coords - seed_coords) @ rot_mat
    rotated_seed_coords = np.array([0, 0, 0])

    # calculate squared distances normalized by the squared semi-axis lengths
    x_dist_norm = (rotated_cloud[:, 0] / a) ** 2
    y_dist_norm = (rotated_cloud[:, 1] / b) ** 2
    z_dist_norm = (rotated_cloud[:, 2] / b) ** 2

    neighbor_ids = np.where(x_dist_norm + y_dist_norm + z_dist_norm <= 1)[0]

    neighbor_ids = neighbor_ids[neighbor_ids != seed_id]

    neighbor_ids = cloud.iloc[neighbor_ids].index

    return neighbor_ids.tolist()


def neighbors_ellipsoid(cloud, seed_id, config):
    """
    find neighbors of seed_id within an ellipsoidal shape.
    """
    seed_data = cloud.iloc[seed_id]
    coordinates_seed = seed_data[['x', 'y', 'z']].values
    coordinates_cloud = cloud[['x', 'y', 'z']].values

    a = config.local_features.supernormal_ellipsoid_a
    b = config.local_features.supernormal_ellipsoid_bc  # Semi-minor axis along y and z

    # Calculate squared distances normalized by the squared semi-axis lengths
    x_dist_norm = ((coordinates_cloud[:, 0] - coordinates_seed[0]) / a) ** 2
    y_dist_norm = ((coordinates_cloud[:, 1] - coordinates_seed[1]) / b) ** 2
    z_dist_norm = ((coordinates_cloud[:, 2] - coordinates_seed[2]) / b) ** 2

    # Sum the normalized distances and find indices where the sum is less than or equal to 1
    neighbor_ids = np.where(x_dist_norm + y_dist_norm + z_dist_norm <= 1)[0]

    # Remove seed_id from neighbor_ids to exclude the point itself
    neighbor_ids = neighbor_ids[neighbor_ids != seed_id]

    return neighbor_ids


def neighbors_oriented_cylinder(cloud, seed_id, config):
    """
    find neighbors of seed_id in cylinder shape
    """
    seed_data = cloud.iloc[seed_id]

    return idx


def neighbors_sphere(cloud_tree, seed_data, config, step=None, radius_in=None):

    if type(seed_data['x']) == np.float64:
        x, y, z = seed_data['x'], seed_data['y'], seed_data['z']
    else:
        x, y, z = seed_data['x'].values[0], seed_data['y'].values[0], seed_data['z'].values[0]

    if step == 'context':
        radius = config.local_neighborhood.context_radius
    else:
        if radius_in is not None:
            radius = radius_in
        else:
            radius = config.supernormal.radius
    idx = cloud_tree.query_ball_point([x, y, z], r=radius)

    return idx



def neighbors_oriented_cuboid(cloud, seed_id, config):
    """
    find neighbors of seed_id in cuboid shape
    """
    seed_data = cloud.iloc[seed_id]

    return idx


def neighbors_aabb_cube(cloud, seed_id, config, step, cluster_lims=None):
    """
    find neighbors of seed_id in axis aligned bounding box shape
    """
    coordinates_cloud = cloud[['x', 'y', 'z', 'id']].values

    if step == 'bbox_mask':
        dist = config.local_neighborhood.radius_a
        # check dist from points to cluster_lims
        # find neighbors that are at least dist away from cluster_lims
        x_ok_lower = np.where(coordinates_cloud[:, 0] > cluster_lims[0] - dist)[0]
        x_ok_upper = np.where(coordinates_cloud[:, 0] < cluster_lims[1] + dist)[0]
        y_ok_lower = np.where(coordinates_cloud[:, 1] > cluster_lims[2] - dist)[0]
        y_ok_upper = np.where(coordinates_cloud[:, 1] < cluster_lims[3] + dist)[0]
        z_ok_lower = np.where(coordinates_cloud[:, 2] > cluster_lims[4] - dist)[0]
        z_ok_upper = np.where(coordinates_cloud[:, 2] < cluster_lims[5] + dist)[0]
        neighbor_ids = np.intersect1d(x_ok_lower, x_ok_upper)
        neighbor_ids = np.intersect1d(neighbor_ids, np.intersect1d(y_ok_lower, y_ok_upper))
        neighbor_ids = np.intersect1d(neighbor_ids, np.intersect1d(z_ok_lower, z_ok_upper))


    else:
        seed_data = cloud.iloc[seed_id]
        coordinates_seed = seed_data[['x', 'y', 'z']].values

        dist = config.local_features.supernormal_cube_dist

        # find neighbors in x direction
        neighbor_ids = np.where(np.abs(coordinates_cloud[:, 0] - coordinates_seed[0]) < dist)[0]
        # find neighbors in y direction
        neighbor_ids = np.intersect1d(neighbor_ids,
                                      np.where(np.abs(coordinates_cloud[:, 1] - coordinates_seed[1]) < dist)[0])
        # find neighbors in z direction
        neighbor_ids = np.intersect1d(neighbor_ids,
                                      np.where(np.abs(coordinates_cloud[:, 2] - coordinates_seed[2]) < dist)[0])

        # remove seed_id from neighbor_ids
        neighbor_ids = neighbor_ids[neighbor_ids != seed_id]

    # retrieve 'id' values from the indices
    neighbor_ids = cloud.iloc[neighbor_ids].index

    return neighbor_ids.tolist()

def patch_context_supernormals(cloud, config):
    patch_ids = np.unique(cloud['ransac_patch'])
    patch_ids = patch_ids[patch_ids != 0]
    # add 'csnx', 'csny', 'csnz' columns to cloud
    cloud['csnx'] = np.nan
    cloud['csny'] = np.nan
    cloud['csnz'] = np.nan
    cloud['csn_confidence'] = np.nan

    for patch_id in patch_ids:
        point_ids = cloud.loc[cloud['ransac_patch'] == patch_id]['id']
        point_ids = point_ids.tolist()
        context_ids, _ = subset_cluster_neighbor_search(cloud, point_ids, config)
        context_normals = cloud.loc[context_ids, ['nx', 'ny', 'nz']].values.astype(np.float32)
        patch_context_sn, sig1, sig2, sig3 = supernormal_svd_s1(context_normals, full_return=True)
        csn_confidence = supernormal_confidence(patch_context_sn, context_normals, sig1, sig2, sig3)

        cloud.loc[cloud['ransac_patch'] == patch_id, 'csnx'] = patch_context_sn[0]
        cloud.loc[cloud['ransac_patch'] == patch_id, 'csny'] = patch_context_sn[1]
        cloud.loc[cloud['ransac_patch'] == patch_id, 'csnz'] = patch_context_sn[2]
        cloud.loc[cloud['ransac_patch'] == patch_id, 'csn_confidence'] = csn_confidence

    return cloud


def subset_cluster_neighbor_search_one_tree(cloud, active_point_ids, radius, full_tree):
    """Optimized approach: Use pre-built KDTree and filter results"""
    active_point_ids = list(active_point_ids)
    # Get bounding box of active points
    active_limits = [
        np.min(cloud.loc[active_point_ids, 'x']),
        np.max(cloud.loc[active_point_ids, 'x']),
        np.min(cloud.loc[active_point_ids, 'y']),
        np.max(cloud.loc[active_point_ids, 'y']),
        np.min(cloud.loc[active_point_ids, 'z']),
        np.max(cloud.loc[active_point_ids, 'z'])
    ]

    # Find neighbors using full tree
    neighbors = []
    for point_id in active_point_ids:
        point = cloud.loc[point_id, ['x', 'y', 'z']].values
        idx = full_tree.query_ball_point(point, radius)
        neighbors.extend(idx)

    # Filter points within bounding box
    neighbor_points = cloud.iloc[list(set(neighbors))]
    mask = ((neighbor_points['x'] >= active_limits[0] - radius) &
            (neighbor_points['x'] <= active_limits[1] + radius) &
            (neighbor_points['y'] >= active_limits[2] - radius) &
            (neighbor_points['y'] <= active_limits[3] + radius) &
            (neighbor_points['z'] >= active_limits[4] - radius) &
            (neighbor_points['z'] <= active_limits[5] + radius))

    reduced_cloud = neighbor_points[mask]
    return reduced_cloud.index.tolist(), reduced_cloud



def subset_cluster_neighbor_search(cloud, active_point_ids, config):
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
            step='patch growing', patch_sn=None, cluster_lims=None, context='context'
        ))
    cluster_neighbors = reduced_cloud.iloc[cluster_neighbors]['id'].tolist()

    return list(set(cluster_neighbors)), reduced_cloud

