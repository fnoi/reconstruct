import random

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
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
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    angles = np.arccos(np.dot(supernormal, normals.T))
    angles = np.rad2deg(angles)
    # angles *= 180 / np.pi

    angles -= 90
    angles = np.abs(angles)

    md_sn = np.median(angles)

    mean_normal = np.mean(normals, axis=0)
    mean_normal /= np.linalg.norm(mean_normal)
    angles = np.arccos(np.dot(mean_normal, normals.T))
    angles = np.rad2deg(angles)
    # median
    md_n = np.median(angles)

    c = 0.1 * len(normals) * md_n / (0.5 * md_sn)

    return c


def angular_deviation(vector, reference):
    vector /= np.linalg.norm(vector)
    reference /= np.linalg.norm(reference)
    angle = np.arccos(np.dot(vector, reference))
    return angle * 180 / np.pi


def calculate_supernormals_rev(cloud=None, cloud_o3d=None, config=None):
    plot_ind = random.randint(0, len(cloud))
    plot_flag = False

    point_ids = np.arange(len(cloud))

    for seed_id in tqdm(point_ids, desc="computing supernormals", total=len(point_ids)):
        seed_data = cloud.iloc[seed_id]

        neighbor_ids = neighborhood_search(cloud, seed_id, config)

        neighbor_normals = cloud.iloc[neighbor_ids][['nx', 'ny', 'nz']].values
        seed_supernormal = supernormal_svd(neighbor_normals)
        seed_supernormal /= np.linalg.norm(seed_supernormal)
        # consistent direction for supernormals
        seed_supernormal = seed_supernormal * np.sign(seed_supernormal[0])
        seed_confidence = supernormal_confidence(seed_supernormal, neighbor_normals)

        cloud.loc[seed_id, 'snx'] = seed_supernormal[0]
        cloud.loc[seed_id, 'sny'] = seed_supernormal[1]
        cloud.loc[seed_id, 'snz'] = seed_supernormal[2]
        cloud.loc[seed_id, 'confidence'] = seed_confidence

        if plot_flag and seed_id == plot_ind:
            plot_patch(cloud_frame=cloud, seed_id=seed_id, neighbor_ids=neighbor_ids)

    return cloud


def ransac_patches(cloud, config):
    print(f'ransac patching')
    cloud['ransac_patch'] = 0
    mask_remaining = np.ones(len(cloud), dtype=bool)
    progress = tqdm()
    blanks = config.clustering.ransac_blanks

    label_id = 0
    while True:
        o3d_cloud_current = o3d.geometry.PointCloud()
        o3d_cloud_current.points = o3d.utility.Vector3dVector(cloud.loc[mask_remaining, ['x', 'y', 'z']].values)

        # ransac plane fitting
        ransac_plane, ransac_inliers = o3d_cloud_current.segment_plane(
            distance_threshold=config.clustering.ransac_dist_thresh,
            ransac_n=config.clustering.ransac_n,
            num_iterations=config.clustering.ransac_iterations
        )
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
            mask_remaining[external_cluster_idx] = False

        progress.update()

    debug_plot = False
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





def region_growing_rev(cloud, config):
    mask_remaining = np.ones(len(cloud), dtype=bool)
    progress = tqdm()
    label_id = 0
    while True:  # loop until no more points are left (thresh)
        # find seed point

        while True:  # loop until segment is complete / region stops growing
            # add patch for all (added) neighbors
            a = 0
            # find cluster (!) neighbors


def neighborhood_search(cloud, seed_id, config):
    seed_data = cloud.iloc[seed_id]
    match config.local_features.neighbor_shape:
        case "sphere":
            cloud_tree = KDTree(cloud[['x', 'y', 'z']].values)
            neighbor_ids = cloud_tree.query_ball_point([seed_data['x'], seed_data['y'], seed_data['z']],
                                                       r=config.local_features.supernormal_radius)
        case "cylinder":
            neighbor_ids = neighbors_oriented_cylinder(cloud, seed_id, config)
        case "cuboid":
            neighbor_ids = neighbors_oriented_cuboid(cloud, seed_id, config)
        case "ellipsoid":
            neighbor_ids = neighbors_oriented_ellipsoid(cloud, seed_id, config)
        case _:
            raise ValueError(f'neighborhood shape "{config.local_features.neighbor_shape}" not implemented')

    return neighbor_ids


def neighbors_oriented_ellipsoid(cloud, seed_id, config):
    """
    find neighbors of seed_id in ellipsoid shape
    """
    seed_data = cloud.iloc[seed_id]

    return idx


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

