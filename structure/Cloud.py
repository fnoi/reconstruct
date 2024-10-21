import copy
import itertools
import os
import time

from tools.fitting_nsga import solve_w_nsga
from tools.metrics import huber_loss
from tools.visual import transformation_tracer

try:
    import numpy as np
    import open3d as o3d
    import pandas as pd
    import pyransac3d as pyrsc
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    from pyswarm import pso
    from sklearn.cluster import DBSCAN

    from tools.IO import points2txt, lines2obj, cache_meta
    from tools.fitting_1 import params2verts
    from tools.fitting_pso import plot_2D_points_bbox, cost_fct_1, cs_plot
    from tools import geometry as geom
    from tools import visual as vis, fitting_pso
except ImportError as e:
    print(f'Import Error: {e}')


class Segment(object):
    def __init__(self, name: str = None, config=None):
        self.left_3D = None
        self.right_3D = None
        self.center_3D = None
        self.vector_3D = None
        self.left_2D = None
        self.right_2D = None
        self.center_2D = None

        self.rotation_pose = None
        self.rotation_long = None
        self.translation = None
        self.transformation_matrix = None

        self.h_beam_verts = None
        self.break_flag = None
        self.angle_2D = None
        self.points_2D = None
        self.points_2D_fitting = None
        self.line_cog_center = None
        self.line_cog_right = None
        self.line_cog_left = None
        self.cog_2D = None

        self.h_beam_params = None
        self.h_beam_params_lookup = None

        self.points_2D = None
        self.points_data = None
        self.points_cleaned = None
        self.intermediate_points = []

        self.left_edit = False
        self.left_joint = False
        self.right_edit = False
        self.right_joint = False
        self.radius = None

        self.name = name
        self.points_center = None
        self.points = None
        self.points = None
        self.parent_path = f'data/out/'
        self.outpath = f'data/out/{name}'

        self.config = config

        # check if directory name exists
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        else:
            # delete files in directory
            files = os.listdir(self.outpath)
            for file in files:
                os.remove(f'{self.outpath}/{file}')

    def load_from_df(self, df, name: str):
        label = name.split('_')[1]
        data = df[df['instance_pr'] == int(label)]
        self.points_data = data
        self.points = data[['x', 'y', 'z']].values
        # self.points = data[:, :3]
        self.points_center = np.mean(self.points, axis=0)

    def load_from_txt(self, name: str):
        path = f'data/in/{name}.txt'
        with open(path, 'r') as f:
            data = f.readlines()
            data = [line.strip().split(' ') for line in data]
            data = np.array(data, dtype=np.float32)

        self.points = data[:, :3]
        self.points_center = np.mean(self.points, axis=0)


    def calc_axes(self, plot=True):
        """
        calculate the principal axes of the segment (core + overpowered function, consider modularizing)
        """
        # insertion
        plot = True
        self.points_hom = np.hstack((self.points, np.ones((self.points.shape[0], 1))))
        points = self.points
        try:
            normals = self.points_data[['nx', 'ny', 'nz']].values
        except TypeError:  # normals inavailable if called in skeleton aggregation
            # calculate normals real quick
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.preprocess.normals_radius, max_nn=self.config.preprocess.normals_max_nn))
            normals = np.asarray(pcd.normals)

        # find the two best planes and their line of intersection
        planes, direction, origin, inliers_0, inliers_1 = geom.orientation_estimation(
            np.concatenate((points, normals), axis=1),
            config=self.config,
            step="skeleton"
        )

        if planes is None:
            self.break_flag = True
            return

        n_0 = np.array(planes[0][:3], dtype=np.float64)  # normal of plane 0

        points_on_line, closest_ind = geom.project_points_to_line(points, origin, direction)

        ref_x = -100000
        ref_t = (ref_x - origin[0]) / direction[0]
        ref_pt = origin + ref_t * direction
        vecs = points_on_line - ref_pt
        dists = np.linalg.norm(vecs, axis=1)
        l_ind = np.argmin(dists)
        r_ind = np.argmax(dists)

        # raw line data from points projected to segment direction vector
        self.left_3D = points_on_line[l_ind]
        self.right_3D = points_on_line[r_ind]
        direction = self.right_3D - self.left_3D
        self.vector_3D = direction / np.linalg.norm(direction)
        self.center_3D = (self.left_3D + self.right_3D) / 2

        self.translation = self.left_3D
        self.transformation_matrix = np.eye(4)
        self.transformation_matrix[:3,3] = self.translation

        # find projection plane and lines indicating the planes (2D)
        proj_plane = geom.normal_and_point_to_plane(self.vector_3D, self.left_3D)

        proj_dir_0, proj_origin_0 = geom.intersecting_line(proj_plane, planes[0])
        proj_dir_1, proj_origin_1 = geom.intersecting_line(proj_plane, planes[1])

        len_proj = self.config.skeleton_visual.line_length_projection
        proj_dir_0 = np.array(
            [[self.left_3D[0] - (proj_dir_0[0] * len_proj),
              self.left_3D[1] - (proj_dir_0[1] * len_proj),
              self.left_3D[2] - (proj_dir_0[2] * len_proj)],
             [self.left_3D[0] + (proj_dir_0[0] * len_proj),
              self.left_3D[1] + (proj_dir_0[1] * len_proj),
              self.left_3D[2] + (proj_dir_0[2] * len_proj)]])
        proj_dir_1 = np.array(
            [[self.left_3D[0] - (proj_dir_1[0] * len_proj),
              self.left_3D[1] - (proj_dir_1[1] * len_proj),
              self.left_3D[2] - (proj_dir_1[2] * len_proj)],
             [self.left_3D[0] + (proj_dir_1[0] * len_proj),
              self.left_3D[1] + (proj_dir_1[1] * len_proj),
              self.left_3D[2] + (proj_dir_1[2] * len_proj)]])
        proj_lines = [proj_dir_0, proj_dir_1]

        proj_points_plane = geom.points_to_actual_plane(points, self.vector_3D, self.left_3D)

        origin = np.array([0, 0, 0])
        target_axis_s = np.array([0, 0, 1])  # Z AXIS
        target_axis_n = np.array([0, 1, 0])  # Y AXIS

        # calc length between self.left and self.right
        length = np.linalg.norm(self.right_3D - self.left_3D)
        target_left = target_axis_n
        target_center = origin
        target_right = target_axis_s
        target = (target_left, target_center, target_right)

        source_left = self.left_3D + n_0
        source_center = self.left_3D
        source_right = source_center + self.vector_3D
        source = (source_left, source_center, source_right)
        self.transformation_matrix = geom.simplified_transform_lines(source, target)

        self.rotation_pose = geom.rotation_matrix_from_vectors(self.vector_3D, target_axis_s)
        proj_points_flat = np.dot(proj_points_plane, self.rotation_pose)

        target_points = np.dot(self.transformation_matrix, self.points_hom.T).T[:,:3]
        transformation_tracer(self.points, target_points, source_angle=source, target_angle=target)

        self.points_2D = target_points[:, :2]

        ransac_data = (inliers_0, inliers_1)
        plot = False
        if plot:
            vis.segment_projection_2D(self.points_2D, ransac_highlight=True, ransac_data=ransac_data)

            # vis.segment_projection_3D(points, proj_lines)
            # vis.segment_projection_3D(proj_points_plane, proj_lines)



    def update_axes(self):
        """bring back center of gravity cog (from lookup) to its correct position in the original coordinate system"""
        cog_flat = rotate_points_2D(self.cog_2D, - self.angle_2D)

        cog_lifted = np.array([cog_flat[0], cog_flat[1], self.z_delta])

        cog_left_maybe = np.dot(cog_lifted, self.mat_rotation_xy)

        projected, _ = project_points_to_line(
            points=self.points,
            point_on_line=cog_left_maybe,
            direction=self.line_raw_dir
        )

        ref_x = -100000
        ref_t = (ref_x - cog_left_maybe[0]) / self.line_raw_dir[0]
        ref_pt = cog_left_maybe + ref_t * self.line_raw_dir
        vecs = projected - ref_pt
        dists = np.linalg.norm(vecs, axis=1)
        l_ind = np.argmin(dists)
        r_ind = np.argmax(dists)

        self.line_cog_left = projected[l_ind]
        self.line_cog_right = projected[r_ind]
        self.line_cog_center = (self.line_cog_left + self.line_cog_right) / 2

    def find_cylinder(self):
        cyl = pyrsc.Cylinder()
        res = cyl.fit(self.points, 0.04, 1000)

        return res

    def calc_pca_o3d(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        # clean up point cloud to improve pca results
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=.75)
        self.points_cleaned = np.asarray(pcd_clean.points)

        cov = pcd_clean.compute_mean_and_covariance()
        pc = np.linalg.eig(cov[1])

        self.pca = pc[1][:, 0]
        self.pcb = pc[1][:, 1]
        self.pcc = pc[1][:, 2]


    def fit_cs_rev(self, config=None):
        points_after_sampling = config.cs_fit.n_downsample
        # plot_2D_points_bbox(self.points_2D)
        # self.downsample_dbscan_grid(config.cs_fit.grid_size, points_after_sampling)


        # self.downsample_dbscan_rand(points_after_sampling)  # TODO: check method limitations, mitigate risk, investigate weighting
        # plot_2D_points_bbox(self.points_2D_fitting)

        self.points_2D_fitting = self.points_2D
        # timer = time.time()
        self.h_beam_params, self.h_beam_verts, self.h_beam_fit_cost, self.cstype = solve_w_nsga(self.points_2D_fitting, config, self.points_2D)
        # self.h_beam_params, self.h_beam_verts, self.h_beam_fit_cost = fitting_pso.fitting_fct(self.points_2D_fitting)

        cog_x = (self.h_beam_verts[11][0] + self.h_beam_verts[0][0]) / 2
        cog_y = (self.h_beam_verts[5][1] + self.h_beam_verts[0][1]) / 2
        self.left_2D = np.array((cog_x, cog_y))

        # TODO: calculate offset of new CS to old CS / origin. transform to depict left_3D in original CS


        cog_2D_hom = np.array([cog_x, cog_y, 0, 1])
        print(self.left_3D)
        self.left_3D = np.dot(self.transformation_matrix.T, cog_2D_hom.T).T[:3]
        print(self.left_3D)


    def downsample_dbscan_rand(self, points_after_sampling):
        init_count = self.points_2D.shape[0]
        points = self.points_2D
        if points.shape[0] > points_after_sampling * 3:
            points = points[np.random.choice(points.shape[0], points_after_sampling, replace=False)]

        eps = 0.05
        min_samples = int(0.05 * points_after_sampling)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_

        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        filtered_points = points[core_samples_mask]

        if filtered_points.shape[0] > points_after_sampling:
            filtered_points = filtered_points[np.random.choice(filtered_points.shape[0], points_after_sampling, replace=True)]

        self.points_2D_fitting = filtered_points

        print(f'downsampling from {init_count} to {filtered_points.shape[0]} points')

    def downsample_dbscan_grid(self, resolution, points_after_sampling):
        init_count = self.points_2D.shape[0]
        points = self.points_2D
        if points.shape[0] > points_after_sampling * 1.5:
            points = points[np.random.choice(points.shape[0], points_after_sampling, replace=False)]

        eps = 0.05
        min_samples = int(0.05 * points_after_sampling)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_

        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        filtered_points = points[core_samples_mask]

        x = np.arange(filtered_points[:, 0].min(), filtered_points[:, 0].max(), resolution)
        y = np.arange(filtered_points[:, 1].min(), filtered_points[:, 1].max(), resolution)
        xx, yy = np.meshgrid(x, y)
        grid = np.array([xx.flatten(), yy.flatten()]).T

        # count points in grid cells
        grid_count = np.zeros(len(grid))
        for i, point in enumerate(grid):
            grid_count[i] = np.sum(np.all(np.isclose(self.points_2D, point, atol=resolution), axis=1))

        # for each grid cell with points calculate the mean
        grid_mean = []
        for i, point in enumerate(grid):
            if grid_count[i] > 0:
                grid_mean.append(np.mean(self.points_2D[np.all(np.isclose(self.points_2D, point, atol=resolution), axis=1)], axis=0))
        filtered_points = np.array(grid_mean)

        if filtered_points.shape[0] > points_after_sampling:
            filtered_points = filtered_points[np.random.choice(filtered_points.shape[0], points_after_sampling, replace=True)]

        self.points_2D_fitting = filtered_points

        print(f'downsampling from {init_count} to {filtered_points.shape[0]} points')

    def cs_lookup(self, path=None):
        if path is None:
            path = '/data/beams/aisc-shapes-database-v15.0.csv'
            # combine current path with path
            path = os.getcwd() + path
        with open(path, 'r', newline='\n') as f:
            beams = pd.read_csv(f, header=0, sep=';')
            # retrieve name of first column
            uno = beams.columns[0]
            beams_frame = beams[[uno, 'AISC_Manual_Label', 'tw.1', 'tf.1', 'bf.1', 'd.1']]
            # rename columns
            beams_frame.columns = ['type', 'label', 'tw', 'tf', 'bf', 'd']
            # remove all "â€“", replace with nan
            beams_frame = beams_frame.replace('â€“', np.nan, regex=True)
            # replace all , with . for tw tf bf and d
            beams_frame = beams_frame.replace(',', '.', regex=True)
            # drop all rows with –
            beams_frame = beams_frame.replace('–', np.nan, regex=True)
            # convert to numeric in column tw
            beams_frame[['tw', 'tf', 'bf', 'd']] = beams_frame[['tw', 'tf', 'bf', 'd']].apply(pd.to_numeric)
            # beams_frame = beams_frame.apply(pd.to_numeric)

        # divide the selected columns by 1000 to convert to mm
        beams_frame[['tw', 'tf', 'bf', 'd']] = beams_frame[['tw', 'tf', 'bf', 'd']] / 1e3

        # find the closest beam
        tw_fit = self.h_beam_params[2]
        tf_fit = self.h_beam_params[3]
        bf_fit = self.h_beam_params[4]
        d_fit = self.h_beam_params[5]

        # find best fit row in beams_frame with Huber Loss
        delta = 1.0  # TODO: include in config
        beams_frame['HuberLoss'] = (
                beams_frame.apply(lambda row: huber_loss(row['tw'] - tw_fit, delta), axis=1) +
                beams_frame.apply(lambda row: huber_loss(row['tf'] - tf_fit, delta), axis=1) +
                beams_frame.apply(lambda row: huber_loss(row['bf'] - bf_fit, delta), axis=1) +
                beams_frame.apply(lambda row: huber_loss(row['d'] - d_fit, delta), axis=1)
        )
        beams_frame = beams_frame.sort_values(by='HuberLoss', ascending=True)

        # beams_frame['RMSE'] = np.sqrt(
        #     (beams_frame['tw'] - tw_fit) ** 2 +
        #     (beams_frame['tf'] - tf_fit) ** 2 +
        #     (beams_frame['bf'] - bf_fit) ** 2 +
        #     (beams_frame['d'] - d_fit) ** 2)
        # beams_frame = beams_frame.sort_values(by='RMSE', ascending=True)
        tw_lookup = beams_frame['tw'].iloc[0]
        tf_lookup = beams_frame['tf'].iloc[0]
        bf_lookup = beams_frame['bf'].iloc[0]
        d_lookup = beams_frame['d'].iloc[0]
        type_lookup = beams_frame['type'].iloc[0]
        label_lookup = beams_frame['label'].iloc[0]

        improve_xy = False
        if improve_xy:
            # optimize location / improve x0, y0 after param lookup
            delta_xy = 0.01
            range_x = np.linspace(self.h_beam_params[0] - delta_xy,
                                  self.h_beam_params[0] + delta_xy,
                                  10)
            range_y = np.linspace(self.h_beam_params[1] - delta_xy,
                                  self.h_beam_params[1] + delta_xy,
                                  10)
            range_x = np.append(range_x, self.h_beam_params[0])
            range_y = np.append(range_y, self.h_beam_params[1])

            # all combinations of x, y
            xy_combinations = list(itertools.product(range_x, range_y))
            err = np.zeros(len(xy_combinations))
            for i, xy in enumerate(xy_combinations):
                params = [xy[0], xy[1], tw_fit, tf_fit, bf_fit, d_fit]
                err[i] = cost_fct_1(params, self.points_2D, debug_plot=False)

            min_ind = np.argmin(err)
            new_x = xy_combinations[min_ind][0]
            new_y = xy_combinations[min_ind][1]

            # x changed?
            if new_x != self.h_beam_params[0]:
                print(f'changed x from {self.h_beam_params[0]} to {new_x}')
                self.h_beam_params[0] = new_x
            else:
                print(f'x unchanged at {self.h_beam_params[0]}')
            # y changed?
            if new_y != self.h_beam_params[1]:
                print(f'changed y from {self.h_beam_params[1]} to {new_y}')
                self.h_beam_params[1] = new_y
            else:
                print(f'y unchanged at {self.h_beam_params[1]}')

            # plot cs with points
            cs_plot(self.h_beam_verts, self.points_2D)
            # plot new cs
            self.h_beam_verts = params2verts(self.h_beam_params)
            cs_plot(self.h_beam_verts, self.points_2D)

        cog_2D_lookup = (
            self.h_beam_params[0] + bf_lookup / 2,
            self.h_beam_params[1] + d_lookup / 2
        )

        self.cog_2D = cog_2D_lookup
        self.cog_3D = 0     # TODO FIX

        delta_d = self.h_beam_params[5] - d_lookup
        delta_bf = self.h_beam_params[4] - bf_lookup

        self.h_beam_params[0] = self.h_beam_params[0] + delta_bf / 2
        self.h_beam_params[1] = self.h_beam_params[1] + delta_d / 2

        self.h_beam_params = { # TODO: double check overwrite correctness... also it is messy af to not preserve structure
            'x0': self.h_beam_params[0],
            'y0': self.h_beam_params[1],
            'type': type_lookup,
            'label': label_lookup,
            'tw': tw_lookup,
            'tf': tf_lookup,
            'bf': bf_lookup,
            'd': d_lookup
        }

        self.h_beam_verts = params2verts(
            [self.h_beam_params['x0'],
             self.h_beam_params['y0'],
             self.h_beam_params['tw'],
             self.h_beam_params['tf'],
             self.h_beam_params['bf'],
             self.h_beam_params['d']]
        )

        # TODO: identify bf and d deltas and move x0/y0 accordingly (to avoid the overall movement)

        fitting_pso.cs_plot(self.h_beam_verts, self.points_2D)

        return
        # return beams_frame
