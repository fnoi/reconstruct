import copy
import subprocess

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import open3d as o3d

from omegaconf import OmegaConf

from scipy.spatial import KDTree
from tqdm import tqdm

from seg2skeleton import inst2skeleton
from tools.region_growing import region_growing_main
# from tools.clustering import region_growing
from tools.code_cache import region_growing_rev
from tools.IO import cache_io, config_io
from tools.local import calculate_supernormals_rev, ransac_patches, patch_context_supernormals
from tools.metrics import calculate_metrics, supernormal_evaluation, calculate_purity
from tools.model_eval import model_evaluation
# from tools.to_model import model_builder

if __name__ == '__main__':
    config = OmegaConf.load('config_1.yaml')
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
    # 6: cross-sections
    # 7: skeleton refinement
    # 8: model generation
    # 9: model evaluation

    ##########
    ##########
    cache_flag = 3
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
            if config.data.cloud_path.endswith('initial_1.txt'):
                cloud = pd.DataFrame(cloud, columns=['x', 'y', 'z', 'instance_gt'])
            elif config.data.cloud_path.endswith('initial_2.txt'):
                cloud = pd.DataFrame(cloud, columns=['x', 'y', 'z', 'instance_gt', 'nx', 'ny', 'nz'])
            elif config.data.cloud_path.endswith('initial_3.txt'):
                cloud = pd.DataFrame(cloud, columns=['x', 'y', 'z', 'instance_gt', 'nx', 'ny', 'nz'])
            elif config.data.cloud_path.endswith('full_30.txt') or config.data.cloud_path.endswith('full_30_1.txt'):
                cloud = pd.DataFrame(cloud, columns=['x', 'y', 'z', 'instance_gt'])
            cloud['instance_gt'] = cloud['instance_gt'].astype(int)
        del f

        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud[['x', 'y', 'z']].values.astype(np.float32))
        if 'nx' in cloud.columns:
            o3d_cloud.normals = o3d.utility.Vector3dVector(cloud[['nx', 'ny', 'nz']].values.astype(np.float32))

        if config.preprocess.downsample_flag:
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

        # if no normals, estimate and store to df
        if 'nx' not in cloud.columns:
            o3d_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.preprocess.normals_radius, max_nn=config.preprocess.normals_max_nn))
            normals = np.asarray(o3d_cloud.normals)
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
        # store x-y-z-confidence to txt
        cloud_store = copy.deepcopy(cloud)
        # drop the rest first
        cloud_store = cloud_store[['x', 'y', 'z', 'confidence']]
        cloud_store.to_csv(f'{config.project.parking_path}/cloud_supernormals_confidence.txt', sep=' ', index=False)

        cloud_store_2 = copy.deepcopy(cloud)
        cloud_store_2 = cloud_store_2[['x', 'y', 'z', 'csn_confidence']]
        cloud_store_2.to_csv(f'{config.project.parking_path}/cloud_supernormals_confidence_csn.txt', sep=' ', index=False)

        if single_step:
            raise ValueError('stop here, single step')

    if cache_flag <= 3:
        # region growing
        print('\n- compute instance predictions through region growing, report metrics')
        with open(f'{config.project.parking_path}/cache_cloud_2.pickle', 'rb') as f:
            cloud = pd.read_pickle(f)
        del f
        cloud = region_growing_main(cloud, config)
        # scatter plot cloud with instance_gt and instance_pred next to each other
        fig = plt.Figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(cloud['x'], cloud['y'], cloud['z'], c=cloud['instance_gt'], cmap='tab20')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(cloud['x'], cloud['y'], cloud['z'], c=cloud['instance_pr'], cmap='tab20')
        plt.show()
        # cloud = region_growing_rev(cloud, config)
        # cloud = region_growing(cloud, config)
        calculate_metrics(cloud, base='cloud')

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

        metrics_report = True
        if metrics_report:
            calculate_metrics(df_cloud=cloud, base='skeleton', skeleton=skeleton, store=False)
        if single_step:
            raise ValueError('stop here, single step')

    if cache_flag <= 5:
        # skeleton: aggregate, compute metrics
        skeleton = pd.read_pickle(f'{config.project.parking_path}/skeleton_cache.pickle')

        print('\n- skeleton segment aggregation, metrics revised')  # baseline exists but omg indeed

        skeleton.aggregate_bones()
        skeleton.update_bone_ids()

        metrics_report = True
        if metrics_report:
            with open(f'{config.project.parking_path}/cache_cloud_3.pickle', 'rb') as f:
                cloud = pd.read_pickle(f)
            calculate_metrics(df_cloud=cloud, base='skeleton', skeleton=skeleton, store=False)

        skeleton.cache_pickle(config.project.parking_path)
        skeleton.plot_cog_skeleton(headline='skeleton aggregation')
        if single_step:
            raise ValueError('stop here, single step')

    if cache_flag <= 6:
        print('\n- fit cross-sections from catalog')
        # fit cross-sections
        skeleton = pd.read_pickle(f'{config.project.parking_path}/skeleton_cache.pickle')
        for bone in skeleton.bones:
            # export_3D = f'{config.project.parking_path}/YP/{bone.name}_dump_3D.txt'
            # export_2D = f'{config.project.parking_path}/YP/{bone.name}_dump_2D.txt'
            # # export_orientation = f'{config.project.parking_path}/bone_{bone.id}_dump_orientation.txt'
            # with open(export_3D, 'w') as f:
            #     # write bone.points_3D to txt
            #     for point in bone.points:
            #         f.write(f'{point[0]} {point[1]} {point[2]}\n')
            # with open(export_2D, 'w') as f:
            #     # write bone.points_2D to txt
            #     for point in bone.points_2D:
            #         f.write(f'{point[0]} {point[1]}\n')
            # continue

            # try:
            bone.fit_cs_rev(config)
            # except ValueError as e:
            #     print(f'error: {e}')
            #     bone.h_beam_params = False
            #     bone.h_beam_verts = False

        skeleton.plot_cog_skeleton()
        skeleton.cache_pickle(config.project.parking_path)
        skeleton.cache_json(config.project.parking_path)
        if single_step:
            raise ValueError('stop here, single step')

    if cache_flag <= 7:
        # skeleton refinement
        print('\n- skeleton bones join on passing')
        skeleton = pd.read_pickle(f'{config.project.parking_path}/skeleton_cache.pickle')
        for bone in skeleton.bones:
            if bone.h_beam_verts is None:
                raise ValueError('no cross-sections available, stop here')
        skeleton.plot_cog_skeleton()
        skeleton.join_on_passing_v2()
        skeleton.plot_cog_skeleton()

        skeleton.cache_pickle(config.project.parking_path)
        skeleton.cache_json(config.project.parking_path)
        if single_step:
            raise ValueError('stop here, single step')

    if cache_flag <= 8:
        raise ValueError('stop here, anyway')  # currently blender works but not integrated
        meth = 'blender'
        if meth == 'blender':
            # run two scripts subsequently in blender
            print('\n- model generation, not gonna happen')
        elif meth == 'ios':
            with open('/Users/fnoic/PycharmProjects/reconstruct/data/parking/skeleton_cache.json', 'r') as f:
                skeleton = pd.read_json(f)
            model_builder(skeleton, config)
        else:
            raise ValueError('method not found')
        if single_step:
            raise ValueError('stop here, single step')


    if cache_flag <= 9:
        print(f' - model evaluation')
        # load skeleton
        skeleton = pd.read_pickle(f'{config.project.parking_path}/skeleton_cache.pickle')

        model_evaluation(skeleton)

