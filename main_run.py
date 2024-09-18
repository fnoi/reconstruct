import os
import pathlib

import numpy as np
import pandas as pd
import open3d as o3d

from omegaconf import OmegaConf

from scipy.spatial import KDTree
from tqdm import tqdm

from seg2skeleton import inst2skeleton
from tools.clustering import region_growing
from tools.code_cache import region_growing_rev
from tools.IO import cache_io, config_io
from tools.local import calculate_supernormals_rev, ransac_patches, patch_context_supernormals
from tools.metrics import calculate_metrics, supernormal_evaluation, calculate_purity

if __name__ == '__main__':
    config = OmegaConf.load('config_experiment_2.yaml')
    config = config_io(config)

    ##########
    ##########
    # cache_flag defines starting point
    # 0: from scratch, preprocessing
    # 1: planar patches
    # 2: supernormals
    # 3: region growing
    # 4: skeleton initiation
    # 5: skeleton aggregation
    # 6: cross-section fitting
    # 7: skeleton refinement
    # ((8: model generation))
    ##########
    ##########
    cache_flag = 4
    single_step = False
    ##########
    ##########

    if cache_flag == 0:
        print('\n- from scratch, preprocessing')
        with open(config.project.path, 'r') as f:
            # TODO: add option to load rgb here, currently XYZ, label only
            cloud = pd.read_csv(f, sep=' ', header=None).values
            # cloud = pd.DataFrame(cloud, columns=['x', 'y', 'z', 'old_label', 'instance_gt'])
            # cloud.drop(['old_label'], axis=1, inplace=True)
            cloud = pd.DataFrame(cloud, columns=['x', 'y', 'z', 'instance_gt'])
            cloud['instance_gt'] = cloud['instance_gt'].astype(int)
        del f

        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud[['x', 'y', 'z']].values.astype(np.float32))
        downsample = True
        if downsample:
            print(f'original cloud size: {len(cloud)}')
            cloud_o3d = o3d_cloud.voxel_down_sample(voxel_size=config.preprocess.voxel_size)
            cloud_frame_new = pd.DataFrame(np.asarray(cloud_o3d.points), columns=['x', 'y', 'z'])
            # add instance_gt column with nan
            cloud_frame_new['instance_gt'] = np.nan
            for row in tqdm(cloud_frame_new.iterrows(), total=len(cloud_frame_new),
                            desc='re-assigning instance_gt to downsampled cloud'):
                point_coords = np.asarray(row[1])[0:3]
                # calculate distance to all points in cloud
                dists = np.linalg.norm(cloud[['x', 'y', 'z']].values - point_coords, axis=1)
                # find the closest point
                seed_id = np.argmin(dists)
                # assign instance_gt to new cloud_frame
                cloud_frame_new.loc[row[0], 'instance_gt'] = cloud.loc[seed_id, 'instance_gt']
            cloud = cloud_frame_new
            print(f'downsampled cloud size: {len(cloud)}')
        cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.preprocess.normals_radius, max_nn=config.preprocess.normals_max_nn))
        normals = np.asarray(cloud_o3d.normals)
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        cloud['nx'] = normals[:, 0]
        cloud['ny'] = normals[:, 1]
        cloud['nz'] = normals[:, 2]

        if 'id' not in cloud.columns:
            # add column to cloud that stores integer ID
            cloud['id'] = [i for i in range(len(cloud))]

        cache_io(cloud=cloud, path=config.project.parking_path, cache_flag=0)
        if single_step:
            raise ValueError('stop here, single step')

    if cache_flag <= 1:
        # compute planar patches
        print('\n- compute planar patches')
        with open(f'{config.project.parking_path}/cache_cloud_0.pickle', 'rb') as f:
            cloud = pd.read_pickle(f)
        del f

        cloud = ransac_patches(cloud, config)
        cloud = patch_context_supernormals(cloud, config)
        # cloud = planar_patches(cloud, config)
        # report mean patch size
        patch_sizes = cloud.groupby('ransac_patch').size()
        print(f'mean patch size: {patch_sizes.mean()}')
        print(f'median patch size: {patch_sizes.median()}')
        cache_io(cloud=cloud, path=config.project.parking_path, cache_flag=1)

        report_purity = True
        if report_purity:
            planar_patch_ids = cloud['ransac_patch'].unique()
            purity = calculate_purity(gt=cloud['instance_gt'], pred=cloud['ransac_patch'], )

        # store to x,y,z,ransac_id
        cloud.to_csv(f'{config.project.parking_path}/cloud_ransac_patches.txt', sep=' ', index=False)
        if single_step:
            raise ValueError('stop here, single step')

        # # optional quality check
        # control_normals = True
        # if control_normals:
        #     normal_evaluation(cloud, config)

    if cache_flag <= 2:
        # compute supernormals
        print('\n- compute supernormals')
        with open(f'{config.project.parking_path}/cache_cloud_1.pickle', 'rb') as f:
            cloud = pd.read_pickle(f)
        del f

        cloud_tree = KDTree(cloud[['x', 'y', 'z']].values.astype(np.float32))
        cloud = calculate_supernormals_rev(cloud, cloud_tree, config)

        cache_io(cloud=cloud, path=config.project.parking_path, cache_flag=2)

        # optional quality check
        control_supernormals = True
        if control_supernormals:
            cloud = supernormal_evaluation(cloud, config, inplace=True)
            cache_io(cloud=cloud, path=config.project.parking_path, cache_flag=2)
        o3d_cloud_0 = o3d.geometry.PointCloud()
        o3d_cloud_0.points = o3d.utility.Vector3dVector(cloud[['x', 'y', 'z']].values.astype(np.float32))
        o3d_cloud_0.normals = o3d.utility.Vector3dVector(cloud[['nx', 'ny', 'nz']].values.astype(np.float32))
        o3d.io.write_point_cloud(f'{config.project.parking_path}/cloud_normals_trial.ply', o3d_cloud_0)
        # create o3d cloud
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud[['x', 'y', 'z']].values.astype(np.float32))
        cloud_o3d.normals = o3d.utility.Vector3dVector(cloud[['snx', 'sny', 'snz']].values.astype(np.float32))
        # save to trial.ply in parking
        o3d.io.write_point_cloud(f'{config.project.parking_path}/cloud_supernormals_trial.ply', cloud_o3d)
        print(f'saved to {config.project.parking_path}/cloud_supernormals_trial.ply')
        if single_step:
            raise ValueError('stop here, single step')

    if cache_flag <= 3:
        # region growing
        print('\n- compute instance predictions through region growing, report metrics')
        with open(f'{config.project.parking_path}/cache_cloud_2.pickle', 'rb') as f:
            cloud = pd.read_pickle(f)
        del f
        cloud = region_growing_rev(cloud, config)
        # cloud = region_growing(cloud, config)
        miou_weighted, miou_unweighted = calculate_metrics(cloud, base='cloud')

        cache_io(cloud=cloud, path=config.project.parking_path, cache_flag=3)
        # store cloud to  .txt
        cloud.to_csv(f'{config.project.parking_path}/cloud_instance_predictions_rev.txt', sep=' ', index=False)
        if single_step:
            raise ValueError('stop here, single step')

    if cache_flag <= 4:
        # skeleton: initiate
        print('\n- initiate skeleton, aggregate skeleton (incl. orientation and projection)')
        with open(f'{config.project.parking_path}/cache_cloud_3.pickle', 'rb') as f:
            cloud = pd.read_pickle(f)
        skeleton = inst2skeleton(cloud, config, df_cloud_flag=True, plot=False)
        skeleton.cache_pickle(config.project.parking_path)
        skeleton.plot_cog_skeleton(headline='skeleton initiation')

        # TODO: are we still retrieving from table?
        if single_step:
            raise ValueError('stop here, single step')

    if cache_flag <= 5:
        # skeleton: aggregate, compute metrics
        skeleton = pd.read_pickle(f'{config.project.parking_path}/skeleton_cache.pickle')

        print('\n- skeleton segment aggregation, metrics revised')  # baseline exists but omg indeed

        skeleton.aggregate_bones()

        metrics_report = True
        if metrics_report:
            with open(f'{config.project.parking_path}/cache_cloud_3.pickle', 'rb') as f:
                cloud = pd.read_pickle(f)
            miou_weighted, miou_unweighted = calculate_metrics(df_cloud=cloud, base='skeleton', skeleton=skeleton)

        skeleton.plot_cog_skeleton(headline='skeleton aggregation')
        if single_step:
            raise ValueError('stop here, single step')

    if cache_flag <= 6:
        print('\n- fit cross-sections, lookup cross-sections')
        # fit cross-sections
        skeleton = pd.read_pickle(f'{config.project.parking_path}/skeleton_cache.pickle')
        for bone in skeleton.bones:
            a = 0
            try:
                bone.fit_cs_rev()
                bone.cs_lookup()
            except ValueError as e:
                print(f'error: {e}')
                bone.h_beam_params = False
                bone.h_beam_verts = False

        skeleton.cache_pickle(config.project.parking_path)
        if single_step:
            raise ValueError('stop here, single step')

    if cache_flag <= 7:
        # skeleton refinement
        print('\n- skeleton bones join on passing')
        skeleton = pd.read_pickle(f'{config.project.parking_path}/skeleton_cache.pickle')
        for bone in skeleton.bones:
            if not bone.h_beam_verts:
                raise ValueError('no cross-sections available, stop here')
        skeleton.plot_cog_skeleton()
        skeleton.join_on_passing_v2()
        skeleton.plot_cog_skeleton()

        skeleton.cache_pickle(config.project.parking_path)
        skeleton.cache_json(config.project.parking_path)
        if single_step:
            raise ValueError('stop here, single step')

    # if cache_flag <= 6:
    #     print('\n- collision-free reconstruction with FreeCAD')
    #     skeleton = pd.read_pickle(f'{config.project.parking_path}/skeleton_cache.pickle')
    #     a = 0
