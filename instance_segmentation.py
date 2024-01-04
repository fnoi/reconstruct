import pathlib
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objs as go
from omegaconf import OmegaConf
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from tqdm import tqdm

from tools.local import supernormal_svd, supernormal_confidence, angular_deviation
from tools.utils import find_random_id


def get_neighbors(point_id, full_cloud, dist):
    coords_point_check = cloud_c_n_sn_co[point_id, :3]
    coords_points_all = cloud_c_n_sn_co[:, :3]
    distances = cdist(coords_point_check[None, :], coords_points_all)
    ids_neighbors = np.where(distances < dist)[1]
    return ids_neighbors.tolist()


def get_neighbors_flex(point_ids, full_cloud, dist):
    coords_points_check = cloud_c_n_sn_co[point_ids, :3]
    coords_points_all = cloud_c_n_sn_co[:, :3]
    distances = cdist(coords_points_check, coords_points_all)
    ids_neighbors = np.where(distances < dist)[1]
    return ids_neighbors.tolist()


def region_growth(feature_cloud_c_n_sn_co, config, feature_cloud_tree):
    points_region = []
    # start with high confidence (low value)
    start_id = np.argmin(cloud_c_n_sn_co[:, -1])
    points_active = [start_id]

    for point_check in points_active:
        print(f'point_check {point_check}')
        check_idx = get_neighbors(point_id=point_check, full_cloud=cloud_c_n_sn_co, dist=config.clustering.max_dist_euc)
        for check_id in check_idx:
            # print(f'check_id {check_id}')
            supernormal_deviation = angular_deviation(cloud_c_n_sn_co[check_id, 6:9], cloud_c_n_sn_co[point_check, 6:9])
            if supernormal_deviation < config.clustering.angle_thresh_supernormal:
                points_region.append(check_id)
            else:
                # check distance to all points in the region
                dists = []
                if len(points_region) > 0:
                    for point_region in points_region:
                        dist = np.linalg.norm(cloud_c_n_sn_co[check_id, :3] - cloud_c_n_sn_co[point_region, :3])
                        dists.append(dist)
                    # sort distances and ids
                    dists = np.array(dists)
                    ids = np.argsort(dists)
                    a = 0
    b = 0



    return None


def calc_supernormals(point_coords_arr, point_normals_arr, point_ids_all, point_cloud_tree):
    supernormals = []
    confidences = []

    plot_id = random.randint(0, len(point_ids_all))

    for id_seed in tqdm(point_ids_all, desc='computing super normals', total=len(point_ids_all)):
        point_seed = point_coords_arr[id_seed]

        ids_neighbors = point_cloud_tree.query_ball_point(point_seed, r=config.local.supernormal_radius)
        normals_neighbors = point_normals_arr[ids_neighbors]
        # normalize normals
        normals_neighbors /= np.linalg.norm(normals_neighbors, axis=1)[:, None] * 5
        normal_seed = point_normals_arr[id_seed] / np.linalg.norm(point_normals_arr[id_seed])
        normals_patch = np.concatenate((normals_neighbors, normal_seed[None, :]), axis=0)

        coords_neighbors = point_coords_arr[ids_neighbors]
        arrowheads = coords_neighbors + normals_neighbors

        supernormal = supernormal_svd(normals_patch)
        confidence = supernormal_confidence(supernormal, normals_patch)
        confidences.append(confidence)

        supernormals.append(supernormal)
        supernormal /= np.linalg.norm(supernormal) * 5

        plot_flag = False
        if id_seed == plot_id:
            plot_flag = True
        if plot_flag:
            fig = go.Figure()

            # Add scatter plot for the points
            fig.add_trace(go.Scatter3d(
                x=point_coords_arr[ids_neighbors, 0],
                y=point_coords_arr[ids_neighbors, 1],
                z=point_coords_arr[ids_neighbors, 2],
                mode='markers',
                # marker type +
                marker=dict(
                    size=4,
                    color='grey',
                    opacity=1  # ,
                    # symbol='dot'
                )
            ))

            # Add lines for the normals
            for i in range(len(coords_neighbors)):
                # Start point of each line (origin)
                x0, y0, z0 = coords_neighbors[i, :]
                # End point of each line (origin + normal)
                x1, y1, z1 = coords_neighbors[i, :] + normals_neighbors[i, :]

                fig.add_trace(go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode='lines',  # Set to lines mode
                    line=dict(color='blue', width=0.5)  # Adjust line color and width as needed
                ))

            # add red thick line for supernormal from seed coord
            x0, y0, z0 = point_seed
            x1, y1, z1 = point_seed + supernormal
            fig.add_trace(go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode='lines',  # Set to lines mode
                line=dict(color='red', width=4)  # Adjust line color and width as needed
            ))
            # headline
            fig.update_layout(
                title_text=f'Point {id_seed} with {len(ids_neighbors)} neighbors, confidence {confidence:.2f}')

            # Show the figure
            fig.show()

    # supernormal abs for each axis
    supernormals = np.abs(np.array(supernormals))

    return supernormals, confidences


if __name__ == "__main__":
    config = OmegaConf.load('config.yaml')

    # local runs only, add docker support
    if os.name == 'nt':
        basepath = config.general.basepath_windows
    else:  # os.name == 'posix':
        basepath = config.general.basepath_macos
    path = pathlib.Path(f'{basepath}{config.general.project_path}{config.segmentation.cloud_path}')

    # read point cloud from file to numpy array
    with open(path, 'r') as f:
        point_coords_arr = np.loadtxt(f)[:, :3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_coords_arr)
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=config.local.normals_radius, max_nn=config.local.max_nn))

    point_normals_arr = np.asarray(point_cloud.normals)
    # store point cloud to pcd file
    o3d.io.write_point_cloud(f'{basepath}{config.general.project_path}data/parking/normals.ply', point_cloud)

    point_ids_all = [i for i in range(point_coords_arr.shape[0])]
    point_ids_done = []
    point_cloud_tree = KDTree(point_coords_arr)

    supernormals, confidences = calc_supernormals(point_coords_arr, point_normals_arr, point_ids_all, point_cloud_tree)

    # concat with original point cloud
    cloud_c_sn = np.concatenate((point_coords_arr, supernormals), axis=1)
    cloud_c_n_sn_co = np.concatenate((point_coords_arr, point_normals_arr, supernormals), axis=1)
    cloud_c_n_sn_co = np.concatenate((cloud_c_n_sn_co, np.array(confidences)[:, None]), axis=1)

    # store in txt
    with open(f'{basepath}{config.general.project_path}data/parking/supernormals.txt', 'w') as f:
        np.savetxt(f, cloud_c_sn, fmt='%.6f')

    cloud_clustered = region_growth(cloud_c_n_sn_co, config, point_cloud_tree)

    a = 0
