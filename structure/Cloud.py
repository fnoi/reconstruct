import copy
import itertools
import os
import time

from tools.fitting_nsga import solve_w_nsga
from tools.metrics import huber_loss

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
    from tools.geometry import rotation_matrix_from_vectors, angle_between_planes, line_of_intersection, \
    project_points_onto_plane, rotate_points_to_xy_plane, normal_and_point_to_plane, \
    intersection_point_of_line_and_plane, points_to_actual_plane, project_points_to_line, intersecting_line, \
    rotate_points_3D, orientation_estimation, intersection_point_of_line_and_plane_rev, orientation_2D, rotate_points_2D, rotate_xy2xyz
    from tools import visual as vis, fitting_pso
except ImportError as e:
    print(f'Import Error: {e}')


class Segment(object):
    def __init__(self, name: str = None, config=None):
        self.h_beam_verts = None
        self.break_flag = None
        self.cog_3D = None
        self.angle_2D = None
        self.points_2D_fitting = None
        self.line_cog_center = None
        self.line_cog_right = None
        self.line_cog_left = None
        self.cog_2D = None
        self.h_beam_params = None
        self.h_beam_params_lookup = None
        self.points_2D = None
        self.points_data = None
        self.line_raw_center = None
        self.line_raw_dir = None
        self.points_cleaned = None
        self.intermediate_points = []
        self.line_raw_left = None
        self.left_edit = False
        self.left_joint = False
        self.line_raw_right = None
        self.right_edit = False
        self.right_joint = False
        self.radius = None
        self.rot_mat_pcb = None
        self.rot_mat_pca = None
        self.pcc = None
        self.pcb = None
        self.name = name
        self.points_center = None
        self.pca = None
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
        planes, direction, origin, inliers_0, inliers_1 = orientation_estimation(
            np.concatenate((points, normals), axis=1),
            config=self.config,
            step="skeleton"
        )
        # print(origin)

        if planes is None:
            self.break_flag = True
            return

        points_on_line, closest_ind = project_points_to_line(points, origin, direction)

        ref_x = -100000
        ref_t = (ref_x - origin[0]) / direction[0]
        ref_pt = origin + ref_t * direction
        vecs = points_on_line - ref_pt
        dists = np.linalg.norm(vecs, axis=1)
        l_ind = np.argmin(dists)
        r_ind = np.argmax(dists)

        # raw line data from points projected to segment direction vector
        self.line_raw_left = points_on_line[l_ind]
        self.line_raw_right = points_on_line[r_ind]
        self.line_raw_dir = direction
        self.line_raw_center = (self.line_raw_left + self.line_raw_right) / 2

        # find projection plane and lines indicating the planes
        proj_plane = normal_and_point_to_plane(self.line_raw_dir, self.line_raw_left)
        self.point = intersection_point_of_line_and_plane_rev(origin, direction, proj_plane)

        proj_dir_0, proj_origin_0 = intersecting_line(proj_plane, planes[0])
        proj_dir_1, proj_origin_1 = intersecting_line(proj_plane, planes[1])

        len_proj = self.config.skeleton_visual.line_length_projection
        proj_dir_0 = np.array(
            [[self.point[0] - (proj_dir_0[0] * len_proj),
              self.point[1] - (proj_dir_0[1] * len_proj),
              self.point[2] - (proj_dir_0[2] * len_proj)],
             [self.point[0] + (proj_dir_0[0] * len_proj),
              self.point[1] + (proj_dir_0[1] * len_proj),
              self.point[2] + (proj_dir_0[2] * len_proj)]])
        proj_dir_1 = np.array(
            [[self.point[0] - (proj_dir_1[0] * len_proj),
              self.point[1] - (proj_dir_1[1] * len_proj),
              self.point[2] - (proj_dir_1[2] * len_proj)],
             [self.point[0] + (proj_dir_1[0] * len_proj),
              self.point[1] + (proj_dir_1[1] * len_proj),
              self.point[2] + (proj_dir_1[2] * len_proj)]])
        proj_lines = [proj_dir_0, proj_dir_1]

        proj_points_plane = points_to_actual_plane(points, self.line_raw_dir, self.line_raw_left)
        proj_origin_plane = points_to_actual_plane(np.array([origin]), self.line_raw_dir, self.line_raw_left)

        proj_points_flat, self.mat_rotation_xy = rotate_points_to_xy_plane(proj_points_plane, self.line_raw_dir)
        proj_origin_flat, _ = rotate_points_to_xy_plane(proj_origin_plane, self.line_raw_dir)

        # move points to z=0 / xy-plane
        self.z_delta = proj_points_flat[0, 2]
        proj_points_flat[:, 2] = proj_points_flat[:, 2] - self.z_delta
        self.points_2D = proj_points_flat[:, :2]
        true_origin_2D = proj_origin_flat[:, 2] - self.z_delta
        true_origin_2D = proj_origin_flat[0, :2]

        proj_origin_flat = proj_origin_flat[0, :2]

        proj_lines_flat = []
        for line in proj_lines:
            line_new, _ = rotate_points_to_xy_plane(line, self.line_raw_dir)
            line_new = line_new[:, :2]
            proj_lines_flat.append(line_new)
        # proj_lines_flat = rotate_points_to_xy_plane(proj_lines, self.line_raw_dir)
        # angle between line plane 1 and x-axis
        line_plane_2D_0 = proj_lines_flat[0][1] - proj_lines_flat[0][0]
        line_plane_2d_1 = proj_lines_flat[1][1] - proj_lines_flat[1][0]

        line_plane_2D_0 = line_plane_2D_0 / np.linalg.norm(line_plane_2D_0)
        line_plane_2D_0 = line_plane_2D_0[:2]
        x_axis = np.array([1, 0])
        angle = np.arccos(np.dot(line_plane_2D_0, x_axis) / (np.linalg.norm(line_plane_2D_0) * np.linalg.norm(x_axis)))
        self.angle_2D = angle
        # rotate points to align line with x-axis

        # vis.segment_projection_2D(proj_points_flat, proj_lines_flat)
        true_origin_2D = rotate_points_2D(true_origin_2D, angle)
        self.points_2D = rotate_points_2D(self.points_2D, angle)
        line_plane_2D_rot_0 = rotate_points_2D(line_plane_2D_0, angle)
        line_plane_2D_rot_1 = rotate_points_2D(line_plane_2d_1, angle)
        lines_plane_fix = [line_plane_2D_rot_0, line_plane_2D_rot_1]

        ransac_data = (inliers_0, inliers_1)
        if plot:
            vis.segment_projection_2D(self.points_2D, lines=lines_plane_fix, extra_point=true_origin_2D,
                                      ransac_highlight=True, ransac_data=ransac_data)

            # vis.segment_projection_3D(points, proj_lines)
            # vis.segment_projection_3D(proj_points_plane, proj_lines)

        # dists = np.linalg.norm(self.points_2D - np.mean(self.points_2D, axis=0), axis=1)
        dists = np.linalg.norm(self.points_2D - np.median(self.points_2D, axis=0), axis=1)
        closest_ind = np.argmin(dists)
        #### TODO

        cog2D_x = np.mean(self.points_2D[:, 0])
        cog2D_y = np.mean(self.points_2D[:, 1])
        self.cog_2D = np.array([cog2D_x, cog2D_y])

        self.cog_3D = rotate_xy2xyz(self.cog_2D, self.mat_rotation_xy, self.angle_2D)

        points_on_line, closest_ind = project_points_to_line(self.points, self.cog_3D, self.line_raw_dir)
        # find left and right points
        ref_x = -99999999
        ref_t = (ref_x - self.cog_3D[0]) / self.line_raw_dir[0]
        ref_pt = self.cog_3D + ref_t * self.line_raw_dir
        vecs = points_on_line - ref_pt
        dists = np.linalg.norm(vecs, axis=1)
        l_ind = np.argmin(dists)
        r_ind = np.argmax(dists)
        self.line_cog_left = points_on_line[l_ind]
        self.line_cog_right = points_on_line[r_ind]
        self.line_cog_center = (self.line_cog_left + self.line_cog_right) / 2


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

    def recompute_pca(self):
        self.pca = self.line_raw_left - self.line_raw_right

    def transform_clean(self):
        """
        calculate projections, rotation matrices, meta information for skeleton
        """

        # move points to origin
        self.points = self.points_cleaned - self.points_center
        # plane point
        plane_point = np.array([0, 0, 0])

        # TODO: here original pca-pcb-pcc can be extracted still

        pc_candidates = [self.pca, self.pcb, self.pcc]
        extent = []
        for pc in pc_candidates:
            points_dist = np.dot(self.points - plane_point, pc)
            left = self.points_center + pc * np.min(points_dist)
            right = self.points_center + pc * np.max(points_dist)
            extent.append(np.linalg.norm(left - right))

        # choose pca with largest extent
        self.pca = pc_candidates[np.argsort(extent)[-1]]
        self.pcb = pc_candidates[np.argsort(extent)[-2]]
        self.pcc = pc_candidates[np.argsort(extent)[-3]]

        # plane normal
        plane_normal = self.pca
        # calculate distance of each point from the plane
        points_dist = np.dot(self.points - plane_point, plane_normal)
        self.line_raw_left = self.points_center + plane_normal * np.min(points_dist)
        self.line_raw_right = self.points_center + plane_normal * np.max(points_dist)
        lines2obj([(self.line_raw_left, self.line_raw_right)], path=self.outpath, topic='skeleton',
                  center=self.points_center)
        # calculate projection of each point on the plane
        points_proj = self.points - np.outer(points_dist, plane_normal)
        # rotation matrix to rotate the plane normal into the z-axis
        z_vec = np.array([0, 0, 1])
        y_vec = np.array([0, 1, 0])
        x_vec = np.array([1, 0, 0])

        # here pcb_rot and pcc_rot needed?
        self.rot_mat_pca = rotation_matrix_from_vectors(plane_normal, x_vec)
        # added transposed because the rotation matrix is on the left of the vector
        pca_rot = np.dot(self.pca, self.rot_mat_pca.T)
        pcb_rot = np.dot(self.pcb, self.rot_mat_pca.T)
        pcc_rot = np.dot(self.pcc, self.rot_mat_pca.T)
        self.rot_mat_pcb = rotation_matrix_from_vectors(pcb_rot, y_vec)
        pcb_rot = np.dot(pcb_rot, self.rot_mat_pcb.T)
        pcc_rot = np.dot(pcc_rot, self.rot_mat_pcb.T)

        # test
        self.rot_mat_test = np.matmul(self.rot_mat_pca, self.rot_mat_pcb)

        # cache_meta(data={'rot_mat_pca': self.rot_mat_pca, 'rot_mat_pcb': self.rot_mat_pcb},
        #            path=self.outpath, topic='rotations')
        lines2obj(lines=[pcb_rot, pcc_rot], path=self.outpath, topic='pcbc')

        # rotate normal vector using the rotation matrix
        normal_rot = np.dot(plane_normal, self.rot_mat_pca)
        normal_rot = np.dot(normal_rot, self.rot_mat_pcb)

        # rotate the points
        points_rot = np.dot(points_proj, self.rot_mat_pca)
        points_rot = np.dot(points_rot, self.rot_mat_pcb)

        pcd_rot = o3d.geometry.PointCloud()
        pcd_rot.points = o3d.utility.Vector3dVector(points_rot)
        cleaned, ind = pcd_rot.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # TODO create new center points in center of gravity of projected points
        # TODO adapt the centerline accordingly with relative translation
        # points2txt(pointset=np.asarray(cleaned.points), path=self.outpath, topic='points_rot')
        points2txt(pointset=points_rot, path=self.outpath, topic='points_flat')

        # plot points in xy iwth scatter
        fig, ax = plt.subplots()
        ax.scatter(points_rot[:, 0], points_rot[:, 1], s=0.1)
        # ax.set_aspect('equal', 'box')
        plt.show()

        return

    def pc2obj(self, pc_type):
        if pc_type == 'initial':
            with open(f'{self.parent_path}/{self.name}_pca_{pc_type}.obj', 'w') as f:
                f.write(f'v {self.points_center[0]} {self.points_center[1]} {self.points_center[2]} \n'
                        f'v {self.pca[0] + self.points_center[0]} {self.pca[1] + self.points_center[1]} {self.pca[2] + self.points_center[2]} \n'
                        f'v {self.pcb[0] + self.points_center[0]} {self.pcb[1] + self.points_center[1]} {self.pcb[2] + self.points_center[2]} \n'
                        f'v {self.pcc[0] + self.points_center[0]} {self.pcc[1] + self.points_center[1]} {self.pcc[2] + self.points_center[2]} \n'
                        f'l 1 2 \n'
                        f'l 1 3 \n'
                        f'l 1 4')
        return

    def fit_cs_rev(self):
        points_after_sampling = 200  # big impact, consider to make it a parameter
        grid_resolution = 0.01
        # plot_2D_points_bbox(self.points_2D)
        # self.downsample_dbscan_grid(grid_resolution, points_after_sampling)
        self.downsample_dbscan_rand(points_after_sampling)  # TODO: check method limitations, mitigate risk, investigate weighting
        # plot_2D_points_bbox(self.points_2D_fitting)

        # timer = time.time()
        self.h_beam_params, self.h_beam_verts, self.h_beam_fit_cost = solve_w_nsga(self.points_2D_fitting)
        # print(f'elapsed time: {time.time() - timer:.3f}')
        # timer = time.time()


        # print(f' working on {self.name}, full point cloud with {self.points_2D.shape[0]} points')
        # solve_me = solve_w_nsga(self.points_2D)

        # print(f'elapsed time: {time.time() - timer:.3f}')

        # raise ValueError("This is a test exception")


        # self.h_beam_params, self.h_beam_verts, self.h_beam_fit_cost = fitting_pso.fitting_fct(self.points_2D_fitting)
        # fitting_pso.cs_plot(self.h_beam_verts, self.points_2D)

        cog_x = (self.h_beam_verts[11][0] + self.h_beam_verts[0][0]) / 2
        cog_y = (self.h_beam_verts[5][1] + self.h_beam_verts[0][1]) / 2
        self.cog_2D = np.array((cog_x, cog_y))

        self.cog_3D = rotate_xy2xyz(self.cog_2D, self.mat_rotation_xy, self.angle_2D)

        # points_on_line, closest_ind = project_points_to_line(self.points, self.cog_3D, self.line_raw_dir)
        # # find left and right points
        # ref_x = -99999999
        # ref_t = (ref_x - self.cog_3D[0]) / self.line_raw_dir[0]
        # ref_pt = self.cog_3D + ref_t * self.line_raw_dir
        # vecs = points_on_line - ref_pt
        # dists = np.linalg.norm(vecs, axis=1)
        # l_ind = np.argmin(dists)
        # r_ind = np.argmax(dists)
        # self.line_cog_left = points_on_line[l_ind]
        # self.line_cog_right = points_on_line[r_ind]
        # self.line_cog_center = (self.line_cog_left + self.line_cog_right) / 2


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
