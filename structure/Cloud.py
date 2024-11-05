import os
from tools.fitting_nsga import solve_w_nsga
from tools.geometry import simplified_transform_lines, kmeans_points_normals_2D
from tools.visual import transformation_tracer, cs_plot

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

from tools.IO import data_from_IFC
from tools import geometry as geom
from tools import visual as vis, fitting_pso


class Segment(object):
    def __init__(self, name: str = None, config=None):
        self.ref_offset_2D = None
        self.normals_hom = None
        self.points_hom = None
        self.filter_backmap = None
        self.filter_weights = None
        self.normals_2D = None
        self.cstype = None

        self.left_3D = None
        self.right_3D = None
        self.center_3D = None
        self.vector_3D = None
        self.left_2D = None
        self.right_2D = None
        self.center_2D = None

        self.source_angle = None
        self.target_angle = None

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
        self.normals = None
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
        if 'nx' in data.columns:
            self.normals = data[['nx', 'ny', 'nz']].values
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
        self.points_hom = np.hstack((self.points, np.ones((self.points.shape[0], 1))))
        self.normals_hom = np.hstack((self.normals, np.zeros((self.normals.shape[0], 1))))
        
        try:
            normals = self.points_data[['nx', 'ny', 'nz']].values
        except TypeError:  # normals unavailable if called in skeleton aggregation
            # calculate normals real quick
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.preprocess.normals_radius, max_nn=self.config.preprocess.normals_max_nn))
            normals = np.asarray(pcd.normals)

        # find the two best planes and their line of intersection
        planes, direction, origin, inliers_0, inliers_1 = geom.orientation_estimation_s2(
            np.concatenate((self.points, normals), axis=1),
            config=self.config,
            step="skeleton"
        )

        if planes is None:
            self.break_flag = True
            return

        n_0 = np.array(planes[0][:3], dtype=np.float64)  # normal of plane 0

        points_on_line, closest_ind = geom.project_points_to_line(self.points, origin, direction)

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

        proj_points_plane = geom.points_to_actual_plane(self.points, self.vector_3D, self.left_3D)

        origin = np.array([0, 0, 0])
        target_axis_s = np.array([0, 0, 1])  # Z AXIS
        target_axis_n = np.array([0, 1, 0])  # Y AXIS

        # calc length between self.left and self.right
        length = np.linalg.norm(self.right_3D - self.left_3D)
        target_left = target_axis_n
        target_center = origin
        target_right = target_axis_s
        self.target_angle = (target_left, target_center, target_right)

        source_left = self.left_3D + n_0
        source_center = self.left_3D
        source_right = source_center + self.vector_3D
        self.source_angle = (source_left, source_center, source_right)
        self.transformation_matrix = geom.simplified_transform_lines(self.source_angle, self.target_angle)

        self.rotation_pose = geom.rotation_matrix_from_vectors(self.vector_3D, target_axis_s)

        target_points = np.dot(self.transformation_matrix, self.points_hom.T).T[:,:3]
        target_normals = np.dot(self.transformation_matrix, self.normals_hom.T).T[:,:3]

        trace = False
        if trace:
            transformation_tracer(self.points, target_points, source_angle=self.source_angle, target_angle=self.target_angle)

        self.points_2D = target_points[:, :2]

        self.normals_2D = target_normals[:, :2] / np.linalg.norm(target_normals[:, :2])

        ransac_data = (inliers_0, inliers_1)
        plot = False
        if plot:
            vis.segment_projection_2D(self.points_2D, ransac_highlight=True, ransac_data=ransac_data)

            # vis.segment_projection_3D(points, proj_lines)
            # vis.segment_projection_3D(proj_points_plane, proj_lines)


    def fit_cs_rev(self, config=None):
        if self.config.cs_fit.n_downsample != 0 and self.config.cs_fit.n_downsample < self.points_2D.shape[0]:
            self.points_2D_fitting, self.normals_2D_fitting, self.filter_weights, self.filter_backmap = kmeans_points_normals_2D(
                self.points_2D, self.normals_2D, self.config.cs_fit.n_downsample)
            # self.downsample_dbscan_rand(config.cs_fit.n_downsample)
        else:
            self.points_2D_fitting = self.points_2D
            self.normals_2D_fitting = self.normals_2D
        # plot_2D_points_bbox(self.points_2D_fitting)
        cs_data, cs_dataframe = data_from_IFC(config.cs_fit.ifc_cs_path)

        if self.config.cs_fit.method == 'nsga3':
            # cross-section fitting with NSGA-III (multi-objective optimization)
            self.h_beam_params, self.h_beam_verts, self.cstype, offset = solve_w_nsga(
                points=self.points_2D_fitting,
                normals=self.normals_2D_fitting,
                config=config,
                all_points=self.points_2D,
                all_normals=self.normals_2D,
                cs_data=cs_data,
                cs_dataframe=cs_dataframe,
                filter_weights=self.filter_weights,
                filter_map=self.filter_backmap
            )
        elif self.config.cs_fit.method == 'pso':
            # cross-section fitting with PSO (single-objective optimization)
            self.h_beam_params, self.h_beam_verts, self.h_beam_fit_cost, offset = fitting_pso.fitting_fct(
                self.points_2D_fitting)
        else:
            # not implemented
            raise NotImplementedError(f'CS fitting method {self.config.cs_fit.method} not implemented')

        self.points_2D = self.points_2D - offset
        self.points_2D_fitting = self.points_2D_fitting - offset
        self.h_beam_verts = self.h_beam_verts - offset
        self.h_beam_params[0] = self.h_beam_params[0] - offset[0]
        self.h_beam_params[1] = self.h_beam_params[1] - offset[1]

        # update COG / left in 2D and 3D
        self.left_3D = geom.calculate_shifted_source_pt(self.source_angle, offset[0], offset[1])
        self.right_3D = geom.calculate_shifted_source_pt(self.source_angle, offset[0], offset[1], third_pt=self.right_3D)
        direction = self.right_3D - self.left_3D
        self.vector_3D = direction / np.linalg.norm(direction)
        self.vector_3D = self.right_3D - self.left_3D
        self.center_3D = (self.left_3D + self.right_3D) / 2

        source_vec_left = self.source_angle[0] - self.source_angle[1]
        source_vec_right = self.source_angle[2] - self.source_angle[1]
        source_vec_center = self.left_3D

        self.source_angle = (source_vec_center + source_vec_left, source_vec_center, source_vec_center + source_vec_right)
        self.transformation_matrix = simplified_transform_lines(self.source_angle, self.target_angle)

        left_3D_hom = np.append(self.left_3D, 1)
        self.left_2D = np.dot(self.transformation_matrix, left_3D_hom)[:2]

        # self.h_beam_verts = params2verts(self.h_beam_params, from_cog=False)
        # if you want to plot here: make sure weights have right size
        cs_plot(vertices=self.h_beam_verts, points=self.points_2D_fitting, normals=self.normals_2D_fitting, weights=self.filter_weights)


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

        # find ids of points in points_2D that are in filtered_points
        ids = []
        for i, point in enumerate(self.points_2D):
            if np.any(np.all(np.isclose(filtered_points, point), axis=1)):
                ids.append(i)

        self.normals_2D = self.normals_2D[ids] # fix downstream issue for plots after NSGA etx

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

