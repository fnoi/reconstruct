import pathlib
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objs as go
from omegaconf import OmegaConf
from scipy.spatial import KDTree
from tqdm import tqdm

from tools.local import supernormal_svd
from tools.utils import find_random_id

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
    supernormals = []

    plot_id = random.randint(0, len(point_ids_all))

    for id_seed in tqdm(point_ids_all, desc='computing super normals', total=len(point_ids_all)):
    # while True:
    #     id_seed = find_random_id(point_ids_done, point_ids_all)
        point_seed = point_coords_arr[id_seed]

        ids_neighbors = point_cloud_tree.query_ball_point(point_seed, r=config.local.max_dist_cross / 3)
        normals_neighbors = point_normals_arr[ids_neighbors]
        # normalize normals
        normals_neighbors /= np.linalg.norm(normals_neighbors, axis=1)[:, None] * 5
        normal_seed = point_normals_arr[id_seed]/ np.linalg.norm(point_normals_arr[id_seed])
        normals_patch = np.concatenate((normals_neighbors, normal_seed[None, :]), axis=0)

        coords_neighbors = point_coords_arr[ids_neighbors]
        arrowheads = coords_neighbors + normals_neighbors

        supernormal = supernormal_svd(normals_patch)
        supernormal /= np.linalg.norm(supernormal) * 5

        supernormals.append(supernormal)

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
                    opacity=1 #,
                    #symbol='dot'
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

            # Show the figure
            fig.show()

    # supernormal abs for each axis
    supernormals = np.abs(np.array(supernormals))

    # concat with original point cloud
    cloud_w_supernormal = np.concatenate((point_coords_arr, supernormals), axis=1)

    # store in txt
    with open(f'{basepath}{config.general.project_path}data/parking/supernormals.txt', 'w') as f:
        np.savetxt(f, cloud_w_supernormal, fmt='%.6f')


    a = 0



