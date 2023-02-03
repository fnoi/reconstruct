import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
#from scipy import stats


class PointCloud(object):
    def __init__(self):
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

        return model, inliers

    def calc_pca_o3d(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        cov = pcd.compute_mean_and_covariance()
        pc = np.linalg.eig(cov[1])

        center = pcd.get_center()
        vec_a = center + pc[1][:, 0]
        vec_b = center + pc[1][:, 1]
        vec_c = center + pc[1][:, 2]

        # write the lines center- vec_a, center- vec_b, center- vec_c to a file in obj format
        with open('data/point_cloud/beam_1_pca.obj', 'w') as f:
            f.write(f'v {center[0]} {center[1]} {center[2]} \n'
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
        # project points to plane using plane normal
        # https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector


