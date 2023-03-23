import copy
import itertools
import os

import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
import matplotlib.pyplot as plt
import sympy as sym

from scipy.spatial import distance
from math import degrees, acos

from pcd_main import plot_cloud

from tools.IO import points2txt, lines2obj, cache_meta
from tools.geometry import rotation_matrix_from_vectors, angle_between_planes, line_of_intersection, \
    project_points_onto_plane, rotate_points_to_xy_plane, normal_and_point_to_plane, \
    intersection_point_of_line_and_plane, points_to_actual_plane, project_points_to_line, intersecting_line, \
    rotate_points


class Segment(object):
    def __init__(self, name: str = None):
        self.dir = None
        self.points_cleaned = None
        self.intermediate_points = []
        self.left = None
        self.left_edit = False
        self.left_joint = False
        self.right = None
        self.right_edit = False
        self.right_joint = False
        self.radius = None
        self.rot_mat_pcb = None
        self.rot_mat_pca = None
        self.pcc = None
        self.pcb = None
        self.name = name
        self.center = None
        self.pca = None
        self.points = None
        self.parent_path = f'data/out/'
        self.outpath = f'data/out/{name}'
        # check if directory name exists
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        else:
            # delete files in directory
            files = os.listdir(self.outpath)
            for file in files:
                os.remove(f'{self.outpath}/{file}')

    def load_from_txt(self, name: str):
        path = f'data/in/{name}.txt'
        with open(path, 'r') as f:
            data = f.readlines()
            data = [line.strip().split(' ') for line in data]
            data = np.array(data, dtype=np.float32)

            self.points = data[:, :3]
            self.center = np.mean(self.points, axis=0)

    def calc_axes(self):
        plane = pyrsc.Plane()
        planes = []
        points = copy.deepcopy(self.points)
        while True:
            res = plane.fit(pts=points, thresh=0.005, minPoints=0.2 * len(points), maxIteration=100000)
            planes.append(res[0])
            points = np.delete(points, res[1], axis=0)
            # calculate angle between planes
            if len(planes) > 1:
                combinations = itertools.combinations(range(len(planes)), 2)
                for comb in combinations:
                    plane1 = planes[comb[0]]
                    plane2 = planes[comb[1]]

                    angle = angle_between_planes(plane1, plane2)
                    if 45 < angle % 180 < 135:  # qualified pair, look no further
                        # point, line = line_of_intersection(plane1, plane2)
                        point, line = intersecting_line(plane1, plane2)
                        # line = line / np.linalg.norm(line)


                        points_on_line = project_points_to_line(self.points, point, line)

                        ref_x = -1000
                        ref_t = (ref_x - point[0]) / line[0]
                        ref_pt = point + ref_t * line
                        vecs = points_on_line - ref_pt
                        dists = np.linalg.norm(vecs, axis=1)
                        l_ind = np.argmin(dists)
                        r_ind = np.argmax(dists)

                        self.left = points_on_line[l_ind]
                        self.right = points_on_line[r_ind]
                        self.dir = line

                        proj_plane = normal_and_point_to_plane(self.dir, self.left)
                        proj_line_pt_0, proj_line_0 = intersecting_line(proj_plane, plane1)
                        proj_line_pt_1, proj_line_1 = intersecting_line(proj_plane, plane2)

                        proj_pts_2 = points_to_actual_plane(self.points, self.dir, self.left)

                        lengthy_boi = 0.2
                        linepts_0_0 = [
                            self.left[0] - proj_line_0[0] * lengthy_boi,
                            self.left[1] - proj_line_0[1] * lengthy_boi,
                            self.left[2] - proj_line_0[2] * lengthy_boi
                        ]
                        linepts_0_1 = [
                            self.left[0] + proj_line_0[0] * lengthy_boi,
                            self.left[1] + proj_line_0[1] * lengthy_boi,
                            self.left[2] + proj_line_0[2] * lengthy_boi
                        ]

                        linepts_1_0 = [
                            self.left[0] - proj_line_1[0] * lengthy_boi,
                            self.left[1] - proj_line_1[1] * lengthy_boi,
                            self.left[2] - proj_line_1[2] * lengthy_boi
                        ]
                        linepts_1_1 = [
                            self.left[0] + proj_line_1[0] * lengthy_boi,
                            self.left[1] + proj_line_1[1] * lengthy_boi,
                            self.left[2] + proj_line_1[2] * lengthy_boi
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

                        ax.scatter(self.left[0], self.left[1], self.left[2], marker='o', s=10)
                        ax.scatter(point[0], point[1], point[2], marker='o', s=10)

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

                        rotated_pts = rotate_points_to_xy_plane(proj_pts_2, self.dir)
                        rotated_linepts = rotate_points_to_xy_plane(np.array([linepts_0_0, linepts_0_1, linepts_1_0, linepts_1_1]), self.dir)

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
                        angle = np.arctan2(rotated_linepts[1, 1] - rotated_linepts[0, 1], rotated_linepts[1, 0] - rotated_linepts[0, 0])
                        # rotate points to align line with x-axis
                        rotated_pivot = rotate_points(np.array([self.left]), angle, np.array([0, 0, 1]))[0]
                        rotated_pts = rotate_points(rotated_pts, angle, np.array([0, 0, 1]))
                        rotated_linepts = rotate_points(rotated_linepts, angle, np.array([0, 0, 1]))

                        if np.mean(rotated_pts[:, 1]) < rotated_pivot[1]:
                            rotated_pts = rotate_points(rotated_pts, np.deg2rad(180), np.array([0, 0, 1]))
                            rotated_linepts = rotate_points(rotated_linepts, np.deg2rad(180), np.array([0, 0, 1]))


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
                        plane_point = intersection_point_of_line_and_plane(point, self.dir, proj_plane)



                        a = 0

                        break
                    else:
                        continue

            if type(self.right) == np.ndarray and type(self.left) == np.ndarray:
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
        self.pca = self.left - self.right

    def transform_clean(self):
        """
        calculate projections, rotation matrices, meta information for skeleton
        """

        # move points to origin
        self.points = self.points_cleaned - self.center
        # plane point
        plane_point = np.array([0, 0, 0])

        # TODO: here original pca-pcb-pcc can be extracted still

        pc_candidates = [self.pca, self.pcb, self.pcc]
        extent = []
        for pc in pc_candidates:
            points_dist = np.dot(self.points - plane_point, pc)
            left = self.center + pc * np.min(points_dist)
            right = self.center + pc * np.max(points_dist)
            extent.append(np.linalg.norm(left - right))

        # choose pca with largest extent
        self.pca = pc_candidates[np.argsort(extent)[-1]]
        self.pcb = pc_candidates[np.argsort(extent)[-2]]
        self.pcc = pc_candidates[np.argsort(extent)[-3]]

        # plane normal
        plane_normal = self.pca
        # calculate distance of each point from the plane
        points_dist = np.dot(self.points - plane_point, plane_normal)
        self.left = self.center + plane_normal * np.min(points_dist)
        self.right = self.center + plane_normal * np.max(points_dist)
        lines2obj([(self.left, self.right)], path=self.outpath, topic='skeleton', center=self.center)
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
                f.write(f'v {self.center[0]} {self.center[1]} {self.center[2]} \n'
                        f'v {self.pca[0] + self.center[0]} {self.pca[1] + self.center[1]} {self.pca[2] + self.center[2]} \n'
                        f'v {self.pcb[0] + self.center[0]} {self.pcb[1] + self.center[1]} {self.pcb[2] + self.center[2]} \n'
                        f'v {self.pcc[0] + self.center[0]} {self.pcc[1] + self.center[1]} {self.pcc[2] + self.center[2]} \n'
                        f'l 1 2 \n'
                        f'l 1 3 \n'
                        f'l 1 4')
        return
