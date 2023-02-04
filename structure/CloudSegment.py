import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
#from scipy import stats


from structure.geometry import rotation_matrix_from_vectors


class CloudSegment(object):
    def __init__(self):
        self.center = None
        self.pca = None
        self.points = None
        self.normals = None
        self.colors = None
        self.features = None

    def load_from_txt(self, file_path: str):
        with open(file_path, 'r') as f:
            data = f.readlines()
            data = [line.strip().split(' ') for line in data]
            data = np.array(data, dtype=np.float32)

            self.points = data[:, :3]
            self.class_id = data[:, 3]
            # self.normals = data[:, 3:6]
            # self.colors = data[:, 6:9]
            # features = data[:, 9:]

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
        vec_a = self.center + pc[1][:, 0]
        vec_b = self.center + pc[1][:, 1]
        vec_c = self.center + pc[1][:, 2]
        
        self.pca = pc[1][:, 0]
        self.pcb = pc[1][:, 1]
        
        # write the lines center- vec_a, center- vec_b, center- vec_c to a file in obj format
        with open('data/point_cloud/beam_1_pca.obj', 'w') as f:
            f.write(f'v {self.center[0]} {self.center[1]} {self.center[2]} \n'
                    f'v {vec_a[0]} {vec_a[1]} {vec_a[2]} \n'
                    f'v {vec_b[0]} {vec_b[1]} {vec_b[2]} \n'
                    f'v {vec_c[0]} {vec_c[1]} {vec_c[2]} \n'
                    f'l 1 2 \n'
                    f'l 1 3 \n'
                    f'l 1 4')



        # with open('data/point_cloud/beam_1_pca.txt', 'w') as f:
        #     f.write(f'{center[0]} {center[1]} {center[2]} \n {vec[0]} {vec[1]} {vec[2]}')


        return pc


    def calc_pca(self):
        """
        Calculates the principal components of the point cloud.
        """
        points = self.points
        points_centered = points - np.mean(points, axis=0)
        cov = np.cov(points_centered, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        eigen_vectors = eigen_vectors.T
        eigen_values = np.real(eigen_values)
        eigen_vectors = np.real(eigen_vectors)
        sort_indices = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[sort_indices]
        eigen_vectors = eigen_vectors[sort_indices]

        pc = np.linalg.eig(cov)

        return eigen_values, eigen_vectors, pc


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
        # rot_mat = np.array([
        #     [np.dot(plane_normal, z_vec), -np.linalg.norm(np.cross(plane_normal, z_vec)), 0],
        #     [np.linalg.norm(np.cross(plane_normal, z_vec)), np.dot(plane_normal, z_vec), 0],
        #     [0, 0, 1]
        # ])
        # rot_mat = np.array([[plane_normal[0], plane_normal[1], plane_normal[2]],
        #                     [plane_normal[1], -plane_normal[0], 0],
        #                     [-plane_normal[2], 0, 1]])
        rot_mat_1 = rotation_matrix_from_vectors(plane_normal, z_vec)
        pcb_rot = np.dot(self.pcb, rot_mat_1)
        rot_mat_2 = rotation_matrix_from_vectors(pcb_rot, y_vec)

        # rotate normal vector using the rotation matrix
        normal_rot = np.dot(plane_normal, rot_mat_1)
        normal_rot = np.dot(normal_rot, rot_mat_2)
        # normal_back
        with open('data/point_cloud/pc_src_rot.obj', 'w') as f:
            f.write(f'v {0} {0} {0} \n'
                    f'v {plane_normal[0]} {plane_normal[1]} {plane_normal[2]} \n'
                    f'v {z_vec[0]} {z_vec[1]} {z_vec[2]} \n'
                    f'v {normal_rot[0]} {normal_rot[1]} {normal_rot[2]} \n'
                    f'l 1 2 \n'
                    f'l 1 3 \n'
                    f'l 1 4')


        # rotate the points
        points_rot = np.dot(points_proj, rot_mat_1)
        points_rot = np.dot(points_rot, rot_mat_2)

        # write the projected points to a file in txt format
        with open('data/point_cloud/points_proj.txt', 'w') as f:
            for i in range(points_proj.shape[0]):
                f.write(f'{points_proj[i][0]} {points_proj[i][1]} {points_proj[i][2]} \n')

        # write the points to a file in txt format
        with open('data/point_cloud/points_rot.txt', 'w') as f:
            for i in range(points_rot.shape[0]):
                f.write(f'{points_rot[i][0]} {points_rot[i][1]} {points_rot[i][2]} \n')

        a = 0


        # plane normal
        n = self.pca
        # plane point
        p = np.array([0, 0, 0])
        # point cloud
        points = self.points
        # calculate distance of each point from the plane
        d = np.dot(points - p, n)
        # calculate projection of each point on the plane
        points_proj = points - np.outer(d, n)
        # write the points to a file in txt format
        with open('data/point_cloud/beam_1_plane.txt', 'w') as f:
            for i in range(points_proj.shape[0]):
                f.write(f'{points_proj[i][0]} {points_proj[i][1]} {points_proj[i][2]} \n')

        # calculate the angle between the pc and the x-y plane
        angle = np.arccos(np.dot(n, np.array([0, 0, 1])))
        # calculate the rotation matrix between the pc and the x-y plane
        rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        # rotate the points to the x-y plane
        points_proj = np.dot(points_proj, rot_mat)


        # rotate the points to the x-y plane


        # points_proj = np.dot(points_proj, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

        with open('data/point_cloud/beam_1_plane_flat.txt', 'w') as f:
            for i in range(points_proj.shape[0]):
                f.write(f'{points_proj[i][0]} {points_proj[i][1]} {points_proj[i][2]} \n')

        return points_proj

