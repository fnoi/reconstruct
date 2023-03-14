import os

import numpy as np
import open3d as o3d

from matplotlib import pyplot as plt


if __name__ == "__main__":
    path = 'C:/Users/ga25mal/PycharmProjects/reconstruct/data/in/'
    # points = []
    # for cloud in os.listdir(path):
    #     if cloud.startswith('beam'):
    #         with open(path + cloud, 'r') as f:
    #             lines = f.readlines()
    #             for line in lines:
    #                 _ = line.split()
    #                 points.append([float(_[0]), float(_[1]), float(_[2])])
    # points_arr = np.array(points, dtype=np.float32)
    # # save points to txt
    # with open(f'{path}/combined.txt', 'w') as f:
    #     np.savetxt(f, points_arr, fmt='%.6f')
    with open(f'{path}/combined.txt', 'r') as f:
        points_arr = np.loadtxt(f)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_arr)
    # calc normals
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    points_arr_normals = np.asarray(pc.normals)
    pointcloud_arr = np.concatenate((points_arr, points_arr_normals), axis=1)
    # for each point find k nearest neighbors patch
    for point in pointcloud_arr:
        # find k nearest neighbors
        k = 30
        kdtree = o3d.geometry.KDTreeFlann(pc)
        [k, idx, _] = kdtree.search_knn_vector_3d()
        # get patch
        patch = pointcloud_arr[idx, :]

#    scatter plot points with iso view
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], s=0.05)
    ax.set_aspect('equal')
    plt.show()

