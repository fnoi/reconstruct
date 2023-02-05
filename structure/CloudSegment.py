import os

import numpy as np
import open3d as o3d
import pyransac3d as pyrsc

from tools.IO import points2txt, pcab2obj
from structure.geometry import rotation_matrix_from_vectors


class CloudSegment(object):
    def __init__(self, name: str = None):
        self.rot_mat_pcb = None
        self.rot_mat_pca = None
        self.pcc = None
        self.pcb = None
        self.name = name
        self.center = None
        self.pca = None
        self.points = None
        self.normals = None
        self.colors = None
        self.features = None
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

        cov = pcd.compute_mean_and_covariance()
        pc = np.linalg.eig(cov[1])

        self.center = pcd.get_center()

        self.pca = pc[1][:, 0]
        self.pcb = pc[1][:, 1]
        self.pcc = pc[1][:, 2]

        pcab2obj(lines=[self.pca, self.pcb, self.pcc], path=self.outpath, topic='pca', center=self.center)

        return pc

    def pts2plane(self):
        """
        calculate projection of 3d points on the x-y plane using the plane normal
        #TODO: store rotation information and extent (-> towards graph)
        """

        # move points to origin
        self.points = self.points - self.center
        # plane normal
        plane_normal = self.pca
        # plane point
        plane_point = np.array([0, 0, 0])
        # calculate distance of each point from the plane
        points_dist = np.dot(self.points - plane_point, plane_normal)
        # calculate projection of each point on the plane
        points_proj = self.points - np.outer(points_dist, plane_normal)
        # rotation matrix to rotate the plane normal into the z-axis
        z_vec = np.array([0, 0, 1])
        y_vec = np.array([0, 1, 0])

        self.rot_mat_pca = rotation_matrix_from_vectors(plane_normal, z_vec)
        pcb_rot_a = np.dot(self.pcb, self.rot_mat_pca)
        pcc_rot_a = np.dot(self.pcc, self.rot_mat_pca)
        self.rot_mat_pcb = rotation_matrix_from_vectors(pcb_rot_a, y_vec)
        pcb_rot_b = np.dot(pcb_rot_a, self.rot_mat_pcb.transpose())
        pcc_rot_b = np.dot(pcc_rot_a, self.rot_mat_pcb.transpose())

        pcab2obj(lines=[pcb_rot_b, pcc_rot_b], path=self.outpath, topic='pcbc')

        # rotate normal vector using the rotation matrix
        normal_rot = np.dot(plane_normal, self.rot_mat_pca)
        normal_rot = np.dot(normal_rot, self.rot_mat_pcb.transpose())

        # rotate the points
        points_rot = np.dot(points_proj, self.rot_mat_pca)
        points_rot = np.dot(points_rot, self.rot_mat_pcb.transpose())

        points2txt(pointset=points_rot, path=self.outpath, topic='points_flat')

        return
