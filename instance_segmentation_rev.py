import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import open3d as o3d
from omegaconf import OmegaConf

from tools.IO import cache_io
from tools.local import calculate_supernormals_rev, region_growing_rev, ransac_patches, neighborhood_plot

if __name__ == '__main__':
    config = OmegaConf.load('config_rev.yaml')
    if os.name == 'nt':
        config.project.path = pathlib.Path(f'{config.project.basepath_windows}{config.project.project_path}{config.segmentation.cloud_path}')
    else:  # os.name == 'posix':
        config.project.path = pathlib.Path(f'{config.project.basepath_macos}{config.project.project_path}{config.segmentation.cloud_path}')
    ##########
    cache_flag = 2  # 0: no cache, 1: load normals, 2: load supernormals and confidence 3: load ransac patches
    ##########

    if cache_flag <= 0:
        print('\n- compute normals')
        with open(config.project.path, 'r') as f:
            # TODO: add option to load rgb here, currently XYZ, label only
            cloud = pd.read_csv(f, sep=' ', header=None).values
            cloud = pd.DataFrame(cloud, columns=['x', 'y', 'z', 'r', 'g', 'b', 'instance_gt'])
            cloud['instance_gt'] = cloud['instance_gt'].astype(int)
            cloud.drop(['r', 'g', 'b'], axis=1, inplace=True)
        del f

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud[['x', 'y', 'z']].values)
        cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.asarray(cloud_o3d.normals)
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        cloud['nx'] = normals[:, 0]
        cloud['ny'] = normals[:, 1]
        cloud['nz'] = normals[:, 2]

        cache_io(xyz=True, normals=True, instance_gt=True,
                 path=config.project.parking_path, cloud=cloud, cache_flag=cache_flag)

    if cache_flag <= 1:
        print('\n- compute supernormals')
        with open(f'{config.project.parking_path}/cache_cloud_0.txt', 'r') as f:
            cloud = pd.read_csv(f, sep=' ', header=None)
            cloud.columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'instance_gt']
        del f

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud[['x', 'y', 'z']].values)
        cloud_o3d.normals = o3d.utility.Vector3dVector(cloud[['nx', 'ny', 'nz']].values)

        cloud['snx'] = None
        cloud['sny'] = None
        cloud['snz'] = None
        cloud['confidence'] = None

        cloud = calculate_supernormals_rev(cloud, cloud_o3d, config)
        # confidence histogram
        # plt.hist(cloud['confidence'], bins=100)
        # plt.show()
        # neighborhood_plot(cloud=cloud, confidence=True)
        # raise ValueError('stop here')


        cache_io(xyz=True, normals=True, supernormals=True, confidence=True, instance_gt=True,
                 path=config.project.parking_path, cloud=cloud, cache_flag=cache_flag)

    if cache_flag <= 2:
        print('\n- compute ransac patches')
        with open(f'{config.project.parking_path}/cache_cloud_1.txt', 'r') as f:
            cloud = pd.read_csv(f, sep=' ', header=None)
            cloud.columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'snx', 'sny', 'snz', 'confidence', 'instance_gt']
        del f

        cloud = ransac_patches(cloud, config)
        cache_io(xyz=True, normals=True, supernormals=True, confidence=True, instance_gt=True, ransac_patch=True,
                    path=config.project.parking_path, cloud=cloud, cache_flag=cache_flag)

    if cache_flag <= 3:
        print('\n- compute instance predictions through region growing')
        with open(f'{config.project.parking_path}/cache_cloud_2.txt', 'r') as f:
            cloud = pd.read_csv(f, sep=' ', header=None)
            cloud.columns = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                             'snx', 'sny', 'snz', 'confidence', 'instance_gt', 'ransac_patch']
        cloud = region_growing_rev(cloud, config)



        a = 0


    a = 0
