import random

import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import KDTree

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
    """
    rev version of supernormal / confidence calculation
    """

    plot_ind = random.randint(0, len(cloud))
    plot_flag = False

    point_ids = np.arange(len(cloud))

    for seed_id in tqdm(point_ids, desc="computing supernormals", total=len(point_ids)):
        seed_data = cloud.iloc[seed_id]

        match config.local_features.neighbor_shape:
            case "sphere":
                cloud_tree = KDTree(cloud[['x', 'y', 'z']].values)
                neighbor_ids = cloud_tree.query_ball_point([seed_data['x'], seed_data['y'], seed_data['z']],
                                                           r=config.local_features.supernormal_radius)
            case "cylinder":
                a = 0
            case "cuboid":
                a = 0
            case "ellipsoid":
                a = 0
            case _:
                raise ValueError("neighborhood shape not implemented")

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


def region_growing_rev(cloud, config):
    """"region growing using ransac and dbscan"""
    ransac_label = 0
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud[['x', 'y', 'z']].values)
    cloud_o3d.normals = o3d.utility.Vector3dVector(cloud[['nx', 'ny', 'nz']].values)
    cloud_tree = KDTree(cloud[['x', 'y', 'z']].values)
    index_mask = np.ones(len(cloud), dtype=bool)
    min_count_current = config.clustering.ransac_min_count

    # ransac plane segmentation
    while True:
        ransac_plane, ransac_inliers = cloud_o3d.segment_plane(
            distance_threshold=config.clustering.ransac_dist_thresh,
            ransac_n=config.clustering.ransac_n,
            num_iterations=config.clustering.ransac_iterations
        )
        if len(ransac_inliers) > config.clustering.ransac_min_count:
            cloud[ransac_inliers, 'ransac_label'] = ransac_label
            ransac_label += 1
            index_mask[ransac_inliers] = False
            # remove points from cloud_o3d
            cloud_o3d = cloud_o3d.select_by_index(np.where(index_mask)[0])

        else:
            min_count_current -= 1
            if min_count_current <= config.clustering.:
                break


