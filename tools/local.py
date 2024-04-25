import copy
import random

import numpy as np
import open3d as o3d
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

from tools.utils import plot_patch


def supernormal_svd(normals):
    U, S, Vt = np.linalg.svd(normals)
    return Vt[-1, :]


def supernormal_confidence(supernormal, normals):
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

    norms = np.linalg.norm(normals, axis=1)[:, None]
    norms = norms + 1e-10  # avoid division by zero
    normals /= norms

    # normals /= np.linalg.norm(normals, axis=1)[:, None]
    angles = np.arccos(np.dot(supernormal, normals.T))
    angles = np.abs(np.rad2deg(angles))
    angles = np.abs(angles - 90)
    angles = np.clip(angles, 0, 90)
    normalized_angles = angles / 90
    c = np.average(1 - normalized_angles)
    c *= len(normals)

    return c


def angular_deviation(vector, reference):
    norm_vector = np.linalg.norm(vector)
    norm_reference = np.linalg.norm(reference)
    if norm_vector == 0 or norm_reference == 0:
        raise ValueError("Input vectors must be non-zero.")

    vector_normalized = vector / norm_vector
    reference_normalized = reference / norm_reference

    # compute dot product and clamp it to the valid range for arccos
    dot_product = np.dot(vector_normalized, reference_normalized)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # compute the angle in radians and then convert to degrees
    angle = np.arccos(dot_product)
    return angle * 180 / np.pi


def neighborhood_calculations(cloud=None, seed_id=None, config=None, plot_ind=None, plot_flag=False):
    neighbor_ids = neighborhood_search(cloud, seed_id, config)

    if plot_flag and seed_id == plot_ind:
        neighborhood_plot(cloud, seed_id, neighbor_ids, config)

    if cloud.loc[seed_id, 'confidence'] is not None:
        # sort neighborhood_ids by confidence
        neighbor_confidences = cloud.loc[neighbor_ids, 'confidence']
        neighbor_ids = neighbor_ids[np.argsort(neighbor_confidences)]
        # choose the worst 80% of the neighborhood
        relative_neighborhood_size = 0.6
        start_index = int(len(neighbor_ids) * (1 - relative_neighborhood_size))
        neighbor_ids = neighbor_ids[start_index:]

    if config.local_features.supernormal_input == 'ransac':
        neighbor_normals = cloud.loc[neighbor_ids][['rnx', 'rny', 'rnz']].values
    else:
        neighbor_normals = cloud.iloc[neighbor_ids][['nx', 'ny', 'nz']].values
    seed_supernormal = supernormal_svd(neighbor_normals)
    seed_supernormal /= np.linalg.norm(seed_supernormal)

    cloud.loc[seed_id, 'snx'] = seed_supernormal[0]
    cloud.loc[seed_id, 'sny'] = seed_supernormal[1]
    cloud.loc[seed_id, 'snz'] = seed_supernormal[2]
    cloud.loc[seed_id, 'confidence'] = supernormal_confidence(seed_supernormal, neighbor_normals)

    return cloud


def calculate_supernormals_rev(cloud=None, config=None):
    cloud['snx'] = None
    cloud['sny'] = None
    cloud['snz'] = None
    cloud['confidence'] = None

    plot_ind = random.randint(0, len(cloud))
    plot_ind = 762
    print(f'plot ind is {plot_ind}')
    plot_flag = True

    point_ids = np.arange(len(cloud))

    if config.local_features.neighbor_shape in ['cube', 'sphere', 'ellipsoid']:  # unoriented neighborhoods
        for seed_id in tqdm(point_ids, desc="computing supernormals, one step", total=len(point_ids)):
                cloud = neighborhood_calculations(cloud=cloud, seed_id=seed_id, config=config,
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
    print(f'ransac patching')
    cloud['rnx'] = 0.0
    cloud['rny'] = 0.0
    cloud['rnz'] = 0.0
    cloud['ransac_patch'] = 0
    mask_remaining = np.ones(len(cloud), dtype=bool)
    progress = tqdm()
    blanks = config.clustering.ransac_blanks

    label_id = 0  # label 0 indicates unclustered
    while True:
        o3d_cloud_current = o3d.geometry.PointCloud()
        o3d_cloud_current.points = o3d.utility.Vector3dVector(cloud.loc[mask_remaining, ['x', 'y', 'z']].values)

        # ransac plane fitting
        ransac_plane, ransac_inliers = o3d_cloud_current.segment_plane(
            distance_threshold=config.clustering.ransac_dist_thresh,
            ransac_n=config.clustering.ransac_n,
            num_iterations=config.clustering.ransac_iterations
        )
        ransac_normal = ransac_plane[0:3]
        inliers_global_idx = np.where(mask_remaining)[0][ransac_inliers]

        # dbscan clustering of inliers
        dbscan_clustering = DBSCAN(
            eps=config.clustering.dbscan_eps_dist,
            min_samples=config.clustering.dbscan_min_count
        ).fit(cloud.loc[inliers_global_idx, ['x', 'y', 'z']].values)
        active_idx = np.where(mask_remaining)[0]

        if len(np.unique(dbscan_clustering.labels_)) == 1:
            blanks -= 1
            if blanks == 0:
                print(f'over, {label_id} patches found')
                break

        for cluster in np.unique(dbscan_clustering.labels_[dbscan_clustering.labels_ != -1]):
            # these points form a valid cluster
            # find the index of the points in the original cloud
            label_id += 1
            cluster_idx = np.where(dbscan_clustering.labels_ == cluster)[0]
            external_cluster_idx = active_idx[ransac_inliers][cluster_idx]
            cloud.loc[external_cluster_idx, 'ransac_patch'] = label_id
            cloud.loc[external_cluster_idx, 'rnx'] = ransac_normal[0]
            cloud.loc[external_cluster_idx, 'rny'] = ransac_normal[1]
            cloud.loc[external_cluster_idx, 'rnz'] = ransac_normal[2]
            mask_remaining[external_cluster_idx] = False

        progress.update()

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


def neighborhood_search(cloud, seed_id, config, step=None, cluster_lims=None, patch_sn=None):

    if step == 'bbox_mask':
        shape = "cube"
    elif step == 'patch growing':
        shape = config.region_growing.neighbor_shape
        seed_data = cloud.iloc[seed_id]
    else:
        shape = config.local_features.neighbor_shape
        seed_data = cloud.iloc[seed_id]
    match shape:
        case "sphere":
            cloud_tree = KDTree(cloud[['x', 'y', 'z']].values)
            neighbor_ids = cloud_tree.query_ball_point([seed_data['x'], seed_data['y'], seed_data['z']],
                                                       r=config.local_features.supernormal_radius)
        case "cylinder":
            neighbor_ids = neighbors_oriented_cylinder(cloud, seed_id, config)
        case "oriented_cuboid":
            neighbor_ids = neighbors_oriented_cuboid(cloud, seed_id, config)
        case "ellipsoid":
            neighbor_ids = neighbors_ellipsoid(cloud, seed_id, config)
        case "cube":
            neighbor_ids = neighbors_aabb_cube(cloud, seed_id, config, step, cluster_lims)
        case "oriented_ellipsoid":
            neighbor_ids = neighbors_oriented_ellipsoid(cloud, seed_id, config, step, patch_sn)
        case "oriented_octahedron":
            neighbor_ids = neighbors_oriented_octahedron(cloud, seed_id, config)  # TODO: add this fct
        case _:
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

    # make a legend for the colors and force the dots larger
    fig.legend(['unclustered', 'cluster_passive', 'cluster_active', 'patch_additions', 'add_candidates', 'angle_checked', 'seed'],
               loc='center left', bbox_to_anchor=(0.4, 0.5))
    # make bigger
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
        match config.local_features.neighbor_shape:
            case "cube":
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
            case "sphere":
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

            case "ellipsoid":
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

            case "oriented_ellipsoid":
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

            case _:
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

    return np.column_stack((direction, y_axis, z_axis))


def neighbors_oriented_ellipsoid(cloud, seed_id, config, step=None, patch_sn=None):
    """
    find neighbors of seed_id in ellipsoid shape
    """
    seed_data = cloud.iloc[seed_id]
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

    return neighbor_ids


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
    coordinates_cloud = cloud[['x', 'y', 'z']].values

    if step == 'bbox_mask':
        dist = config.region_growing.neighborhood_radius_a
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

    return neighbor_ids
