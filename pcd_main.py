import os

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from matplotlib import pyplot as plt

def plot_cloud(pc, head, candidate, c_grav, cross_mean, norms):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.2)
    ax.scatter(candidate[0], candidate[1], candidate[2], s=10, c='g')
    ax.scatter(c_grav[0], c_grav[1], c_grav[2], s=10, c='b')
    ax.plot([candidate[0], c_grav[0]], [candidate[1], c_grav[1]], [candidate[2], c_grav[2]])
    for norm in norms:
        ax.plot(
            [candidate[0], candidate[0] + norm[0]],
            [candidate[1], candidate[1] + norm[1]],
            [candidate[2], candidate[2] + norm[2]],
            c='k', linewidth=0.05
        )
    ax.plot([candidate[0], candidate[0] + cross_mean[0]],
            [candidate[1], candidate[1] + cross_mean[1]],
            [candidate[2], candidate[2] + cross_mean[2]],
            c='r', linewidth=2)
    ax.set_aspect('equal')
    plt.title(head)
    plt.show()


if __name__ == "__main__":
    path = 'C:/Users/ga25mal/PycharmProjects/reconstruct/data/in/'
    path = 'C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/test_beams_1cm.txt'
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
    # with open(f'{path}/combined.txt', 'r') as f:
    with open(path, 'r') as f:
        points_arr = np.loadtxt(f)[:, :3]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_arr[:, :3])
    # calc normals
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    points_arr_normals = np.asarray(pc.normals)
    pointcloud_arr = np.concatenate((points_arr, points_arr_normals), axis=1)
    pointcloud_enc = np.concatenate((points_arr, np.zeros_like(points_arr)), axis=1)


    # for each point find k nearest neighbors patch

    tree = KDTree(pointcloud_arr)
    # k = 30
    # for point in pointcloud_arr:
        # find k nearest neighbors
        # dd, ii = tree.query(point, k=k)
        # get patch
        # patch = pointcloud_arr[ii, :]
        # a = 0

    radius = 0.4
    thresh_dif = 0.8
    patch_idx = tree.query_ball_point(pointcloud_arr, r=radius)

    ok_core_points = []

    # find suitable patch core points

    for pt_id in range(len(pointcloud_arr)):
        print(f'point {pt_id} of {len(pointcloud_arr)}')
        # get patch
        candidate_id = np.random.randint(0, len(pointcloud_arr))
        candidate = pointcloud_arr[candidate_id, :]

        patch = pointcloud_arr[patch_idx[candidate_id], :]
        c_grav = np.mean(patch[:, 0:3], axis=0)
        dist = np.linalg.norm(c_grav - candidate[0:3])
        a = True
        if a:
        # if dist < thresh_dif:
            ok_core_points.append(candidate_id)
            patch_norms = patch[:, 3:6]
            # check if even, if no remove last entry
            if len(patch_norms) % 2 != 0:
                patch_norms = patch_norms[:-1, :]
            # pairwise crossproduct
            cross = np.cross(patch_norms[0::2, :], patch_norms[1::2, :])
            # calc mean of all cross products
            cross_mean = np.mean(cross, axis=0)
            cross_mean = cross_mean / np.linalg.norm(cross_mean)
            # encode patch points with feature
            #pointcloud_arr[candidate_id, 3:6] = cross_mean
            pointcloud_enc[patch_idx[candidate_id], 3:6] = cross_mean

            plot = False
            if plot:
                plot_cloud(pc=patch, head='ok', candidate=candidate, c_grav=c_grav, cross_mean=cross_mean, norms=patch_norms)
            # a = 0
        # else:
        #     a = 0
            # plot_cloud(pc=patch, head='not ok', candidate=candidate, c_grav=c_grav)

    # ok point cloud
    ok_cp = pointcloud_arr[ok_core_points, :]
    with open('C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/test_beams_ok.txt', 'w') as f:
        np.savetxt(f, ok_cp, fmt='%.6f')

    # drop rows from pointcloud_enc where last three columns are zero
    pc_end = pointcloud_enc[~np.all(pointcloud_enc[:, 3:6] == 0, axis=1)]
    with open('C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/test_beams_ok_enc.txt', 'w') as f:
        np.savetxt(f, pc_end, fmt='%.6f')





#    scatter plot points with iso view
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], s=0.05)
    ax.set_aspect('equal')
    plt.show()

    # scatter plot points with iso view
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ok_cp[:, 0], ok_cp[:, 1], ok_cp[:, 2], s=0.05)
    ax.set_aspect('equal')
    plt.show()
