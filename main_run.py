import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import open3d as o3d
import plotly.graph_objects as go

from omegaconf import OmegaConf

from scipy.spatial import KDTree

import tools.utils
from seg2skeleton import inst2skeleton
from tools.clustering import region_growing
from tools.IO import cache_io
from tools.local import calculate_supernormals_rev, ransac_patches, neighborhood_plot, patch_growing, grow_stage_1
from tools.metrics import calculate_metrics, supernormal_evaluation, normal_evaluation

if __name__ == '__main__':
    config = OmegaConf.load('config_full.yaml')
    if os.name == 'nt':
        config.project.path = pathlib.Path(f'{config.project.basepath_windows}{config.project.project_path}{config.segmentation.cloud_path}')
        config.project.orientation_gt_path = pathlib.Path(f'{config.project.basepath_windows}{config.project.project_path}{config.segmentation.orientation_path}')
    else:  # os.name == 'posix':
        config.project.path = pathlib.Path(f'{config.project.basepath_macos}{config.project.project_path}{config.segmentation.cloud_path}')
        config.project.orientation_gt_path = pathlib.Path(f'{config.project.basepath_macos}{config.project.project_path}{config.segmentation.orientation_path}')

    ##########
    cache_flag = 0
    ##########

    if cache_flag <= 1:
        print('\n- compute normals')
        with open(config.project.path, 'r') as f:
            # TODO: add option to load rgb here, currently XYZ, label only
            cloud = pd.read_csv(f, sep=' ', header=None).values
            # cloud = pd.DataFrame(cloud, columns=['x', 'y', 'z', 'old_label', 'instance_gt'])
            # cloud.drop(['old_label'], axis=1, inplace=True)
            cloud = pd.DataFrame(cloud, columns=['x', 'y', 'z', 'instance_gt'])
            cloud['instance_gt'] = cloud['instance_gt'].astype(int)
        del f

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud[['x', 'y', 'z']].values)
        cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.local_features.normals_radius, max_nn=config.local_features.max_nn))
        normals = np.asarray(cloud_o3d.normals)
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        cloud['nx'] = normals[:, 0]
        cloud['ny'] = normals[:, 1]
        cloud['nz'] = normals[:, 2]

        cache_io(cloud=cloud, path=config.project.parking_path, cache_flag=0)

    if cache_flag <= 1:
        print('\n- compute ransac patches')
        with open(f'{config.project.parking_path}/cache_cloud_0.pickle', 'rb') as f:
            cloud = pd.read_pickle(f)
        del f

        cloud = ransac_patches(cloud, config)
        cache_io(cloud=cloud, path=config.project.parking_path, cache_flag=1)

        # optional quality check
        control_normals = True
        if control_normals:
            normal_evaluation(cloud, config)

    if cache_flag <= 2:
        print('\n- compute supernormals')
        with open(f'{config.project.parking_path}/cache_cloud_1.pickle', 'rb') as f:
            cloud = pd.read_pickle(f)
        del f

        cloud_tree = KDTree(cloud[['x', 'y', 'z']].values)
        cloud = calculate_supernormals_rev(cloud, cloud_tree, config)

        cache_io(cloud=cloud, path=config.project.parking_path, cache_flag=2)

        # optional quality check
        control_supernormals = True
        if control_supernormals:
            supernormal_evaluation(cloud, config)
        o3d_cloud_0 = o3d.geometry.PointCloud()
        o3d_cloud_0.points = o3d.utility.Vector3dVector(cloud[['x', 'y', 'z']].values)
        o3d_cloud_0.normals = o3d.utility.Vector3dVector(cloud[['nx', 'ny', 'nz']].values)
        o3d.io.write_point_cloud(f'{config.project.parking_path}/cloud_normals_trial.ply', o3d_cloud_0)
        # create o3d cloud
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud[['x', 'y', 'z']].values)
        cloud_o3d.normals = o3d.utility.Vector3dVector(cloud[['snx', 'sny', 'snz']].values)
        # save to trial.ply in parking
        o3d.io.write_point_cloud(f'{config.project.parking_path}/cloud_supernormals_trial.ply', cloud_o3d)
        print(f'saved to {config.project.parking_path}/cloud_supernormals_trial.ply')
        raise ValueError('stop here')

    if cache_flag <= 3:
        print('\n- compute instance predictions through region growing, report metrics')
        with open(f'{config.project.parking_path}/cache_cloud_2.pickle', 'rb') as f:
            cloud = pd.read_pickle(f)
        del f
        cloud = region_growing(cloud, config)
        miou_weighted, miou_unweighted = calculate_metrics(cloud, config)

        cache_io(cloud=cloud, path=config.project.parking_path, cache_flag=3)
        # store cloud to  .txt
        cloud.to_csv(f'{config.project.parking_path}/cloud_instance_predictions_rev.txt', sep=' ', index=False)
        # raise ValueError('stop here')

    if cache_flag <= 4:
        print('\n- project instance points to plane, initiate skeleton, fit cs')
        with open(f'{config.project.parking_path}/cache_cloud_3.pickle', 'rb') as f:
            cloud = pd.read_pickle(f)
        skeleton = inst2skeleton(cloud, config, df_cloud_flag=True, plot=True)

        # process bones directly
        for bone in skeleton.bones:
            bone.fit_cs_rev()
            print(bone.h_beam_params)
        skeleton.cache_pickle(config.project.parking_path)

    if cache_flag <= 5:
        skeleton = pd.read_pickle(f'{config.project.parking_path}/skeleton_cache.pickle')

        # create plotly fig
        fig = go.Figure()
        for bone in skeleton.bones:
            bone.cs_lookup()
            bone.update_axes()

            # plot line_cog_left, line_cog_right as lines
            fig.add_trace(go.Scatter3d(x=[bone.line_cog_left[0], bone.line_cog_right[0]],
                                       y=[bone.line_cog_left[1], bone.line_cog_right[1]],
                                       z=[bone.line_cog_left[2], bone.line_cog_right[2]],
                                       mode='lines',
                                       line=dict(color='blue', width=3)))
            # add line_cog_left, line_cog_right as scatter points
            fig.add_trace(go.Scatter3d(x=[bone.line_cog_left[0], bone.line_cog_right[0]],
                                       y=[bone.line_cog_left[1], bone.line_cog_right[1]],
                                       z=[bone.line_cog_left[2], bone.line_cog_right[2]],
                                       mode='markers',
                                       marker=dict(color='magenta', size=5)))
            # point cloud scatter
            fig.add_trace(go.Scatter3d(x=bone.points[:, 0],
                                       y=bone.points[:, 1],
                                       z=bone.points[:, 2],
                                       mode='markers',
                                       marker=dict(color='grey', size=1)))

        # perspective should be ortho
        fig.layout.scene.camera.projection.type = "orthographic"
        # no background grid
        fig.layout.scene.xaxis.visible = False
        fig.layout.scene.yaxis.visible = False
        fig.layout.scene.zaxis.visible = False

        # show go figure
        fig.show()

        print('\n- refine skeleton aggregation')  # baseline exists but omg indeed
        # over-aggregate
        # cut by hierarchy (cs dims)
        # join on passing
        # join open ends (stronger dim stays in place? dim and point support?)
        skeleton.join_on_passing()


    if cache_flag <= 6:
        print('\n- collision-free reconstruction with FreeCAD')  # no idea (but should be fine)
