import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import open3d as o3d
from omegaconf import OmegaConf

import tools.utils
from seg2skeleton import inst2skeleton
from tools.clustering import region_growing
from tools.IO import cache_io
from tools.local import calculate_supernormals_rev, ransac_patches, neighborhood_plot, patch_growing, grow_stage_1
from tools.metrics import calculate_metrics, supernormal_evaluation, normal_evaluation

if __name__ == '__main__':
    config = OmegaConf.load('config_rev.yaml')
    if os.name == 'nt':
        config.project.path = pathlib.Path(f'{config.project.basepath_windows}{config.project.project_path}{config.segmentation.cloud_path}')
    else:  # os.name == 'posix':
        config.project.path = pathlib.Path(f'{config.project.basepath_macos}{config.project.project_path}{config.segmentation.cloud_path}')

    ##########
    cache_flag = 4.1
    ##########

    if cache_flag <= 1:
        print('\n- compute normals')
        with open(config.project.path, 'r') as f:
            # TODO: add option to load rgb here, currently XYZ, label only
            cloud = pd.read_csv(f, sep=' ', header=None).values
            cloud = pd.DataFrame(cloud, columns=['x', 'y', 'z', 'old_label', 'instance_gt'])
            cloud['instance_gt'] = cloud['instance_gt'].astype(int)
            cloud.drop(['old_label'], axis=1, inplace=True)
        del f

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud[['x', 'y', 'z']].values)
        cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
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

        cloud = calculate_supernormals_rev(cloud, config)

        cache_io(cloud=cloud, path=config.project.parking_path, cache_flag=2)

        # optional quality check
        control_supernormals = True
        if control_supernormals:
            supernormal_evaluation(cloud, config)

    if cache_flag <= 3:
        print('\n- compute instance predictions through region growing, report metrics')
        with open(f'{config.project.parking_path}/cache_cloud_2.pickle', 'rb') as f:
            cloud = pd.read_pickle(f)
        del f
        cloud = region_growing(cloud, config)
        miou_weighted, miou_unweighted = calculate_metrics(cloud, config)

        cache_io(cloud=cloud, path=config.project.parking_path, cache_flag=3)

    if cache_flag <= 4:
        print('\n- project instance points to plane')  # 1. ok, 2. help 3. easy
        with open(f'{config.project.parking_path}/cache_cloud_3.pickle', 'rb') as f:
            cloud = pd.read_pickle(f)
        del f
        skeleton = inst2skeleton(cloud, config, df_cloud_flag=True)

    if cache_flag <= 4.5:
        print('\n- fit cs')
        with open(f'{config.project.parking_path}/cache.pickle', 'rb') as f:
            data = pd.read_pickle(f)
        a = 0

        for beam in data:
            segment = beam[1]
            name = beam[0]
            # write 2d points to txt
            # np.savetxt(f'/Users/fnoic/Downloads/beam_export_projection_{name}.txt', segment.points_2D, delimiter=' ', fmt='%f')
            segment.fit_cs_rev()



        a = 0


    if cache_flag <= 5:
        print('\n- define initial skeleton and refine by semantics')  # baseline exists but omg

    if cache_flag <= 6:
        print('\n- collision-free reconstruction with FreeCAD')  # no idea (but should be fine)
