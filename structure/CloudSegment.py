import os

import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
import matplotlib.pyplot as plt

from tools.IO import points2txt, lines2obj, cache_meta
from tools.geometry import rotation_matrix_from_vectors


class Segment(object):
    def __init__(self, name: str = None):
        self.points_cleaned = None
        self.intermediate_points = []
        self.left = None
        self.left_edit = False
        self.right = None
        self.right_edit = False
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

    def find_cylinder(self):
        cyl = pyrsc.Cylinder()
        res = cyl.fit(self.points, 0.01, 1000)

        return res

    def calc_pca_o3d(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        # clean up point cloud to improve pca results
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=.75)
        self.points_cleaned = np.asarray(pcd_clean.points)

        cov = pcd_clean.compute_mean_and_covariance()
        pc = np.linalg.eig(cov[1])

        self.center = pcd_clean.get_center()

        self.pca = pc[1][:, 0]
        self.pcb = pc[1][:, 1]
        self.pcc = pc[1][:, 2]

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

        self.rot_mat_pca = rotation_matrix_from_vectors(plane_normal, z_vec)
        pcb_rot = np.dot(self.pcb, self.rot_mat_pca)
        pcc_rot = np.dot(self.pcc, self.rot_mat_pca)
        self.rot_mat_pcb = rotation_matrix_from_vectors(pcb_rot, y_vec)
        pcb_rot = np.dot(pcb_rot, self.rot_mat_pcb)
        pcc_rot = np.dot(pcc_rot, self.rot_mat_pcb)

        cache_meta(data={'rot_mat_pca': self.rot_mat_pca, 'rot_mat_pcb': self.rot_mat_pcb},
                   path=self.outpath, topic='rotations')
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
        #points2txt(pointset=np.asarray(cleaned.points), path=self.outpath, topic='points_rot')
        points2txt(pointset=points_rot, path=self.outpath, topic='points_flat')

        # plot points in xy iwth scatter
        fig, ax = plt.subplots()
        ax.scatter(points_rot[:, 0], points_rot[:, 1], s=0.1)
        ax.set_aspect('equal', 'box')
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
