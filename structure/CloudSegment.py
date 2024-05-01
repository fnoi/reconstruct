import copy
import itertools
import os
import time

import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tools.IO import points2txt, lines2obj, cache_meta
from tools.geometry import rotation_matrix_from_vectors, angle_between_planes, line_of_intersection, \
    project_points_onto_plane, rotate_points_to_xy_plane, normal_and_point_to_plane, \
    intersection_point_of_line_and_plane, points_to_actual_plane, project_points_to_line, intersecting_line, \
    rotate_points_3D, orientation_estimation, intersection_point_of_line_and_plane_rev, orientation_2D, rotate_points_2D
from tools import visual as vis


class Segment(object):
    def __init__(self, name: str = None, config=None):
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

    def calc_axes(self):
        """
        calculate the principal axes of the segment (core + overpowered function, consider modularizing)
        """
        points = self.points
        normals = self.points_data[['nx', 'ny', 'nz']].values
        # find the two best planes and their
        planes, direction, origin, inliers_0, inliers_1 = orientation_estimation(
            np.concatenate((points, normals), axis=1),
            config=self.config,
            step="skeleton"
        )
        # scatter plot and highlight the inlier points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, zorder=1)
        ax.scatter(points[inliers_0, 0], points[inliers_0, 1], points[inliers_0, 2], s=0.5, color='red', zorder=2)
        ax.scatter(points[inliers_1, 0], points[inliers_1, 1], points[inliers_1, 2], s=0.5, color='purple', zorder=3)
        ax.set_aspect('equal')
        fig.show()


        points_on_line = project_points_to_line(points, origin, direction)

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

        len = self.config.skeleton_visualization.line_length_projection
        proj_dir_0 = np.array(
            [[self.point[0] - (proj_dir_0[0] * len),
              self.point[1] - (proj_dir_0[1] * len),
              self.point[2] - (proj_dir_0[2] * len)],
             [self.point[0] + (proj_dir_0[0] * len),
              self.point[1] + (proj_dir_0[1] * len),
              self.point[2] + (proj_dir_0[2] * len)]])
        proj_dir_1 = np.array(
            [[self.point[0] - (proj_dir_1[0] * len),
              self.point[1] - (proj_dir_1[1] * len),
              self.point[2] - (proj_dir_1[2] * len)],
             [self.point[0] + (proj_dir_1[0] * len),
              self.point[1] + (proj_dir_1[1] * len),
              self.point[2] + (proj_dir_1[2] * len)]])
        proj_lines = [proj_dir_0, proj_dir_1]

        proj_points_plane = points_to_actual_plane(points, self.line_raw_dir, self.line_raw_left)
        proj_points_flat, self.mat_rotation_xy = rotate_points_to_xy_plane(proj_points_plane, self.line_raw_dir)
        # move points to z=0, include this in rotation matrix
        self.z_delta = proj_points_flat[0, 2]
        proj_points_flat[:, 2] = proj_points_flat[:, 2] - self.z_delta
        self.points_2D = proj_points_flat[:, :2]

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
        # rotate points to align line with x-axis

        # vis.segment_projection_2D(proj_points_flat, proj_lines_flat)
        self.points_2D = rotate_points_2D(self.points_2D, angle)
        line_plane_2D_rot_0 = rotate_points_2D(line_plane_2D_0, angle)
        line_plane_2D_rot_1 = rotate_points_2D(line_plane_2d_1, angle)
        lines_plane_fix = [line_plane_2D_rot_0, line_plane_2D_rot_1]
        # proj_lines_flat_aligned = rotate_points_2D(proj_lines_flat, angle)

        origin_flat, _ = rotate_points_to_xy_plane(np.array([origin]), self.line_raw_dir)

        ransac_data = (inliers_0, inliers_1)
        vis.segment_projection_2D(self.points_2D, lines=lines_plane_fix, extra_point=origin_flat[0][:2],
                                  ransac_highlight=True, ransac_data=ransac_data)

        # find main orientation in 2D

        vis.segment_projection_3D(points, proj_lines)
        vis.segment_projection_3D(proj_points_plane, proj_lines)
        vis.segment_projection_2D(proj_points_flat, proj_lines_flat, extra_point=origin_flat[0])

        proj_pts_2 = points_to_actual_plane(self.points, self.line_raw_dir, self.line_raw_left)

        lengthy_boi = 0.2
        linepts_0_0 = [
            self.line_raw_left[0] - proj_dir_0[0] * lengthy_boi,
            self.line_raw_left[1] - proj_dir_0[1] * lengthy_boi,
            self.line_raw_left[2] - proj_dir_0[2] * lengthy_boi
        ]
        linepts_0_1 = [
            self.line_raw_left[0] + proj_dir_0[0] * lengthy_boi,
            self.line_raw_left[1] + proj_dir_0[1] * lengthy_boi,
            self.line_raw_left[2] + proj_dir_0[2] * lengthy_boi
        ]
        linepts_1_0 = [
            self.line_raw_left[0] - proj_dir_1[0] * lengthy_boi,
            self.line_raw_left[1] - proj_dir_1[1] * lengthy_boi,
            self.line_raw_left[2] - proj_dir_1[2] * lengthy_boi
        ]
        linepts_1_1 = [
            self.line_raw_left[0] + proj_dir_1[0] * lengthy_boi,
            self.line_raw_left[1] + proj_dir_1[1] * lengthy_boi,
            self.line_raw_left[2] + proj_dir_1[2] * lengthy_boi
        ]

        rotated_pts = rotate_points_to_xy_plane(proj_pts_2, self.line_raw_dir)
        rotated_linepts = rotate_points_to_xy_plane(
            np.array([linepts_0_0, linepts_0_1, linepts_1_0, linepts_1_1]), self.line_raw_dir)

        # calculate angle between line plane 1 and x-axis
        angle = np.arctan2(rotated_linepts[1, 1] - rotated_linepts[0, 1],
                           rotated_linepts[1, 0] - rotated_linepts[0, 0])
        # rotate points to align line with x-axis
        rotated_pivot = rotate_points_3D(np.array([self.line_raw_left]), angle, np.array([0, 0, 1]))[0]
        rotated_pts = rotate_points_3D(rotated_pts, angle, np.array([0, 0, 1]))
        rotated_linepts = rotate_points_3D(rotated_linepts, angle, np.array([0, 0, 1]))

        if np.mean(rotated_pts[:, 1]) < rotated_pivot[1]:
            rotated_pts = rotate_points_3D(rotated_pts, np.deg2rad(180), np.array([0, 0, 1]))
            rotated_linepts = rotate_points_3D(rotated_linepts, np.deg2rad(180), np.array([0, 0, 1]))

        points_proj = rotated_pts

        # points_dist = np.dot(self.points - self.left, self.dir)
        # points_proj = self.points - np.outer(points_dist, self.dir)
        # self.rotmat_main = rotation_matrix_from_vectors(self.dir, np.array([0, 0, 1]))
        # plot scatter points_proj

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(points_proj[:, 0], points_proj[:, 1], s=0.05)
        # get extent of scatter
        mini = np.min(points_proj, axis=0)
        maxi = np.max(points_proj, axis=0)
        # fix ax extent
        ax.set_xlim(mini[0], maxi[0])
        ax.set_ylim(mini[1], maxi[1])
        # find intersecting point on plane
        plane_point = intersection_point_of_line_and_plane(origin, self.line_raw_dir, proj_plane)

        vis.vis_segment_planes_3D(self, origin, planes, proj_plane)
        vis.segment_projection_3D(proj_pts_2, linepts_0_0, linepts_0_1, linepts_1_0, linepts_1_1)
        vis.segment_projection_2D(rotated_pts, rotated_linepts)
        vis.plot_sth_x(rotated_pts, rotated_linepts)

        a = 0

        while True:

            plane_params, plane_inliers = o3d_cloud.segment_plane(
                distance_threshold=self.config.skeleton.ransac_dist_thresh,
                ransac_n=int(round(self.config.skeleton.ransac_min_count_rel * len(points), 0)),
                num_iterations=self.config.skeleton.ransac_iterations)
            end = time.time()

            planes.append(plane_params)
            points = np.delete(points, plane_inliers, axis=0)

            # calculate angle between planes
            if len(planes) > 1:
                combinations = itertools.combinations(range(len(planes)), 2)
                for comb in combinations:
                    plane1 = planes[comb[0]]
                    plane2 = planes[comb[1]]

                    angle = angle_between_planes(plane1, plane2)
                    if 45 < angle % 180 < 135:  # qualified pair, look no further
                        # point, line = line_of_intersection(plane1, plane2)
                        origin, direction = intersecting_line(plane1, plane2)
                        # line = line / np.linalg.norm(line)

                        points_on_line = project_points_to_line(self.points, origin, direction)

                        ref_x = -1000
                        ref_t = (ref_x - origin[0]) / direction[0]
                        ref_pt = origin + ref_t * direction
                        vecs = points_on_line - ref_pt
                        dists = np.linalg.norm(vecs, axis=1)
                        l_ind = np.argmin(dists)
                        r_ind = np.argmax(dists)

                        self.line_raw_left = points_on_line[l_ind]
                        self.line_raw_right = points_on_line[r_ind]
                        self.line_raw_dir = direction

                        proj_plane = normal_and_point_to_plane(self.line_raw_dir, self.line_raw_left)
                        proj_line_pt_0, proj_dir_0 = intersecting_line(proj_plane, plane1)
                        proj_line_pt_1, proj_dir_1 = intersecting_line(proj_plane, plane2)

                        proj_pts_2 = points_to_actual_plane(self.points, self.line_raw_dir, self.line_raw_left)

                        lengthy_boi = 0.2
                        linepts_0_0 = [
                            self.line_raw_left[0] - proj_dir_0[0] * lengthy_boi,
                            self.line_raw_left[1] - proj_dir_0[1] * lengthy_boi,
                            self.line_raw_left[2] - proj_dir_0[2] * lengthy_boi
                        ]
                        linepts_0_1 = [
                            self.line_raw_left[0] + proj_dir_0[0] * lengthy_boi,
                            self.line_raw_left[1] + proj_dir_0[1] * lengthy_boi,
                            self.line_raw_left[2] + proj_dir_0[2] * lengthy_boi
                        ]

                        linepts_1_0 = [
                            self.line_raw_left[0] - proj_dir_1[0] * lengthy_boi,
                            self.line_raw_left[1] - proj_dir_1[1] * lengthy_boi,
                            self.line_raw_left[2] - proj_dir_1[2] * lengthy_boi
                        ]
                        linepts_1_1 = [
                            self.line_raw_left[0] + proj_dir_1[0] * lengthy_boi,
                            self.line_raw_left[1] + proj_dir_1[1] * lengthy_boi,
                            self.line_raw_left[2] + proj_dir_1[2] * lengthy_boi
                        ]

                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], marker='.', s=0.01)
                        thresh = 1
                        xlim = (np.min(self.points[:, 0]) - thresh, np.max(self.points[:, 0]) + thresh)
                        ylim = (np.min(self.points[:, 1]) - thresh, np.max(self.points[:, 1]) + thresh)
                        zlim = (np.min(self.points[:, 2]) - thresh, np.max(self.points[:, 2]) + thresh)

                        x = np.linspace(xlim[0], xlim[1], 3)
                        y = np.linspace(ylim[0], ylim[1], 3)
                        z = np.linspace(zlim[0], zlim[1], 3)

                        x1, y1 = np.meshgrid(x, y)
                        a1, b1, c1, d1 = plane1
                        z1 = (- a1 * x1 - b1 * y1 - d1) / c1
                        ax.plot_surface(x1, y1, z1, alpha=0.3)

                        x2, z2 = np.meshgrid(x, z)
                        a2, b2, c2, d2 = plane2
                        y2 = (- a2 * x2 - c2 * z2 - d2) / b2
                        ax.plot_surface(x2, y2, z2, alpha=0.3)

                        y2, z2 = np.meshgrid(y, z)
                        a2, b2, c2, d2 = proj_plane
                        x2 = (- b2 * y2 - c2 * z2 - d2) / a2
                        ax.plot_surface(x2, y2, z2, alpha=0.3)

                        ax.scatter(self.line_raw_left[0], self.line_raw_left[1], self.line_raw_left[2], marker='o',
                                   s=10)
                        ax.scatter(origin[0], origin[1], origin[2], marker='o', s=10)

                        ax.set_xlim = xlim
                        ax.set_ylim = ylim
                        ax.set_zlim = zlim
                        # ax.set_aspect('equal')
                        fig.show()

                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(proj_pts_2[:, 0], proj_pts_2[:, 1], proj_pts_2[:, 2])
                        # ax.scatter(points_on_line[:, 0], points_on_line[:, 1], points_on_line[:, 2], color='red')
                        ax.plot(
                            [linepts_0_0[0], linepts_0_1[0]],
                            [linepts_0_0[1], linepts_0_1[1]],
                            [linepts_0_0[2], linepts_0_1[2]],
                            color='red')
                        ax.plot(
                            [linepts_1_0[0], linepts_1_1[0]],
                            [linepts_1_0[1], linepts_1_1[1]],
                            [linepts_1_0[2], linepts_1_1[2]],
                            color='purple')
                        ax.set_aspect('equal')
                        fig.show()

                        rotated_pts = rotate_points_to_xy_plane(proj_pts_2, self.line_raw_dir)
                        rotated_linepts = rotate_points_to_xy_plane(
                            np.array([linepts_0_0, linepts_0_1, linepts_1_0, linepts_1_1]), self.line_raw_dir)

                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.scatter(rotated_pts[:, 0], rotated_pts[:, 1], s=0.05)
                        ax.plot(
                            [rotated_linepts[0, 0], rotated_linepts[1, 0]],
                            [rotated_linepts[0, 1], rotated_linepts[1, 1]],
                            color='red')
                        ax.plot(
                            [rotated_linepts[2, 0], rotated_linepts[3, 0]],
                            [rotated_linepts[2, 1], rotated_linepts[3, 1]],
                            color='purple')
                        ax.set_aspect('equal')
                        fig.show()

                        # calculate angle between line plane 1 and x-axis
                        angle = np.arctan2(rotated_linepts[1, 1] - rotated_linepts[0, 1],
                                           rotated_linepts[1, 0] - rotated_linepts[0, 0])
                        # rotate points to align line with x-axis
                        rotated_pivot = rotate_points_3D(np.array([self.line_raw_left]), angle, np.array([0, 0, 1]))[0]
                        rotated_pts = rotate_points_3D(rotated_pts, angle, np.array([0, 0, 1]))
                        rotated_linepts = rotate_points_3D(rotated_linepts, angle, np.array([0, 0, 1]))

                        if np.mean(rotated_pts[:, 1]) < rotated_pivot[1]:
                            rotated_pts = rotate_points_3D(rotated_pts, np.deg2rad(180), np.array([0, 0, 1]))
                            rotated_linepts = rotate_points_3D(rotated_linepts, np.deg2rad(180), np.array([0, 0, 1]))

                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.scatter(rotated_pts[:, 0], rotated_pts[:, 1], s=0.05)
                        ax.plot(
                            [rotated_linepts[0, 0], rotated_linepts[1, 0]],
                            [rotated_linepts[0, 1], rotated_linepts[1, 1]],
                            color='red')
                        ax.plot(
                            [rotated_linepts[2, 0], rotated_linepts[3, 0]],
                            [rotated_linepts[2, 1], rotated_linepts[3, 1]],
                            color='purple')
                        ax.set_aspect('equal')
                        fig.show()

                        points_proj = rotated_pts

                        # points_dist = np.dot(self.points - self.left, self.dir)
                        # points_proj = self.points - np.outer(points_dist, self.dir)
                        # self.rotmat_main = rotation_matrix_from_vectors(self.dir, np.array([0, 0, 1]))
                        # plot scatter points_proj

                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.scatter(points_proj[:, 0], points_proj[:, 1], s=0.05)
                        # get extent of scatter
                        mini = np.min(points_proj, axis=0)
                        maxi = np.max(points_proj, axis=0)
                        # fix ax extent
                        ax.set_xlim(mini[0], maxi[0])
                        ax.set_ylim(mini[1], maxi[1])
                        # find intersecting point on plane
                        plane_point = intersection_point_of_line_and_plane(origin, self.line_raw_dir, proj_plane)

                        a = 0

                        break
                    else:
                        continue

            if type(self.line_raw_right) == np.ndarray and type(self.line_raw_left) == np.ndarray:
                break

        a = 0

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

    def plot_flats(self):
        """
        plot the points in the xy plane in a scatter plot for each segment in subplot
        """
        a = 0

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
