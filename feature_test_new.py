import copy
import os

import numpy as np
import open3d as o3d
import multiprocessing as mp
from scipy.spatial import KDTree
from tqdm import tqdm
from numba import njit, prange

from matplotlib import pyplot as plt



# print(plt.get_backend())


def plot_cloud(pc, head, candidate=False, c_grav=False, cross_mean=False, norms=False, leftright=False):
    if norms:
        norms = norms / 3
    # if cross_mean:
    if type(cross_mean) == np.ndarray:
        cross_mean = cross_mean / 3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.01, c='grey')
    if candidate.any():
        ax.scatter(candidate[0], candidate[1], candidate[2], s=10, c='g')
    if c_grav and candidate.any():
        ax.scatter(c_grav[0], c_grav[1], c_grav[2], s=10, c='b')
        ax.plot([candidate[0], c_grav[0]], [candidate[1], c_grav[1]], [candidate[2], c_grav[2]])
    if norms:
        for norm in norms:
            ax.plot(
                [candidate[0], candidate[0] + norm[0]],
                [candidate[1], candidate[1] + norm[1]],
                [candidate[2], candidate[2] + norm[2]],
                c='k', linewidth=0.05
            )
    # if cross_mean and candidate.any():
    # if cross_mean.any() and candidate.any():
    #     ax.plot([candidate[0], candidate[0] + cross_mean[0]],
    #             [candidate[1], candidate[1] + cross_mean[1]],
    #             [candidate[2], candidate[2] + cross_mean[2]],
    #             c='r', linewidth=2)
    if type(leftright) != bool:
        ax.plot([leftright[0][0], leftright[1][0]],
                [leftright[0][1], leftright[1][1]],
                [leftright[0][2], leftright[1][2]],
                c='r', linewidth=2)
    ax.set_aspect('equal')
    plt.title(head)
    plt.show()

    a = 0


# @njit
# def mean_numba(array):
#     res = []
#     for i in prange(array.shape[1]):
#         res.append(array[:, i].mean())
#
#     return np.array(res)
#
#
# @njit
# def calc_super_normal(patch):
#     patch_normals = patch[:, 3:6]
#     if patch.shape[0] %2 != 0:
#         patch_normals = patch_normals[:-1, :]
#     n = patch_normals.shape[0]
#     crosses = np.cross(patch_normals[:n//2, :], patch_normals[n//2:, :])
#     cross_mean = mean_numba(crosses)
#     # cross_mean = np.mean(crosses, axis=0)
#     cross_mean = cross_mean / np.linalg.norm(cross_mean)
#     return cross_mean


def calc_super_normal_numpy(patch):
    patch_normals = patch[:, 3:6]
    if patch.shape[0] % 2 != 0:
        patch_normals = patch_normals[:-1, :]
    n = patch_normals.shape[0]
    crosses = np.cross(patch_normals[:n//2, :], patch_normals[n//2:, :])
    # cross_mean = mean_numba(crosses)
    cross_mean = np.mean(crosses, axis=0)
    cross_mean = cross_mean / np.linalg.norm(cross_mean)
    return cross_mean


def super_normal_multiproc(patch, n_processes=4):
    with mp.Pool(processes=n_processes) as pool:
        patches = np.array_split(patch, n_processes)
        results = pool.map(calc_super_normal_numpy, patches)
        return np.mean(results, axis=0)


def smooth_features_knn(pointcloud, k=20):
    tree = KDTree(pointcloud[:, :3])
    pointcloud_smooth = np.zeros_like(pointcloud)
    for i in tqdm(range(pointcloud.shape[0]), desc='smoothing features', total=pointcloud.shape[0]):
        point = pointcloud[i, :]
        point_normal = point[3:]
        neighbors = tree.query(point[:3], k=k)[1]
        neighbor_normals = pointcloud[neighbors, 3:]

        neighbor_normals_flipped = np.zeros_like(neighbor_normals)
        for j, neighbor_normal in enumerate(neighbor_normals):
            # if angle between point normal and neighbor normal is > 90°, flip neighbor normal
            if np.dot(point_normal, neighbor_normal) < 0:
                neighbor_normal *= -1
                # print('did it')
            neighbor_normals_flipped[j] = neighbor_normal

        pointcloud_smooth[i, :3] = point[:3]
        pointcloud_smooth[i, 3:] = np.mean(neighbor_normals_flipped, axis=0)
    return pointcloud_smooth


if __name__ == "__main__":
    # path = 'C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/combined_1cm.txt'
    path = 'C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/feature_test_limited.txt'
    plot = False

    with open(path, 'r') as f:
        points_arr = np.loadtxt(f)[:, :3]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_arr[:, :3])
    # calc normals
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    points_arr_normals = np.asarray(pc.normals)
    pointcloud_arr = np.concatenate((points_arr, points_arr_normals), axis=1)
    pointcloud_enc = np.concatenate((points_arr, np.zeros_like(points_arr)), axis=1)

    tree = KDTree(points_arr)

    radius = 0.2
    thresh_dif = 0.5
    thresh_inliers = 10

    # find neighbors within radius using the tree
    patch_idx = tree.query_ball_point(points_arr, r=radius)

    ok_core_points = []

    # find suitable patch core points

    for pt_id in tqdm(range(len(pointcloud_arr)), desc='computing super normals', total=len(pointcloud_arr)):
        # print(f'point {pt_id} of {len(pointcloud_arr)}')
        # get patch
        # candidate_id = np.random.randint(0, len(pointcloud_arr))
        candidate_id = pt_id
        candidate = pointcloud_arr[candidate_id, :]

        patch = pointcloud_arr[patch_idx[candidate_id], :]
        c_grav = np.mean(patch[:, 0:3], axis=0)
        dist = np.linalg.norm(c_grav - candidate[0:3])
        # a = True
        super_normal = True
        if dist < thresh_dif and len(patch) > thresh_inliers:
            # if a:
            ok_core_points.append(candidate_id)
            if super_normal:

                # solution = super_normal_multiproc(patch)
                solution = calc_super_normal_numpy(patch)
                # solution = calc_super_normal(patch)

            else:
                # pick random color from matplotlib tab and save in solution as 1x3 numpy array RGB
                randi = np.random.randint(0, 10)
                solution = np.array(plt.get_cmap('tab10')(randi)[:3])

            cross_mean = solution

            pointcloud_enc[pt_id, 3:6] = cross_mean



            if plot:
                plot_cloud(pc=patch, head='ok', candidate=candidate, c_grav=c_grav, cross_mean=cross_mean,
                           norms=patch_norms)

    pointcloud_enc = smooth_features_knn(pointcloud_enc, k=6)
    pointcloud_end_rgb = copy.deepcopy(pointcloud_enc)
    pointcloud_end_rgb[:, 3:6] = np.abs(pointcloud_end_rgb[:, 3:6]) * 255

    # ok point cloud
    ok_cp = pointcloud_arr[ok_core_points, :]
    with open('C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/test_beams_ok.txt', 'w') as f:
        np.savetxt(f, ok_cp, fmt='%.6f')

    # drop rows from pointcloud_enc where last three columns are zero
    pc_end = pointcloud_enc[~np.all(pointcloud_enc[:, 3:6] == 0, axis=1)]
    with open('C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/test_beams_ok_enc.txt', 'w') as f:
        np.savetxt(f, pc_end, fmt='%.6f')

    pc_end_rgb = pointcloud_end_rgb[~np.all(pointcloud_end_rgb[:, 3:6] == 0, axis=1)]
    with open('C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/test_beams_ok_enc_rgb.txt', 'w') as f:
        np.savetxt(f, pc_end_rgb, fmt='%.6f')

    # scatter plot points with colors
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pc_end_rgb[:, 0], pc_end_rgb[:, 1], pc_end_rgb[:, 2], c=pc_end_rgb[:, 3:6] / 255, s=0.2)
    plt.show()
