import copy
import os

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from tqdm import tqdm

from matplotlib import pyplot as plt


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


if __name__ == "__main__":
    # path = 'C:/Users/ga25mal/PycharmProjects/reconstruct/data/in/'
    # path = 'C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/test_beams_1cm.txt'
    # path = 'C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/test_beams.txt'
    path = 'C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/combined_1cm.txt'
    plot = False
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

    # tree = KDTree(pointcloud_arr)
    tree = KDTree(points_arr)
    # k = 30
    # for point in pointcloud_arr:
    # find k nearest neighbors
    # dd, ii = tree.query(point, k=k)
    # get patch
    # patch = pointcloud_arr[ii, :]
    # a = 0

    radius = 0.3
    thresh_dif = 0.5
    thresh_inliers = 200
    # find neighbors within radius using the tree
    patch_idx = tree.query_ball_point(points_arr, r=radius)
    # patch_idx = tree.query_ball_point(points_arr, r=radius)
    # patch_idx = tree.query(points_arr, distance_upper_bound=radius)
    # patch_idx = tree.query_ball_point(pointcloud_arr, r=radius)

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
                patch_norms = patch[:, 3:6]

                # # v4: qr decomposition
                # A = patch_norms
                # # Compute the QR decomposition of A
                # Q, R = np.linalg.qr(A.T)
                #
                # # Solve the equation A^T x = 0 using QR decomposition
                # # The solution is given by the last column of Q
                # solution = Q[:, -1]

                # check if even, if no remove last entry
                if len(patch_norms) % 2 != 0:
                    patch_norms = patch_norms[:-1, :]
                # pairwise crossproduct
                cross = np.cross(patch_norms[0::2, :], patch_norms[1::2, :])
                # calc mean of all cross products
                cross_mean = np.mean(cross, axis=0)
                cross_mean = cross_mean / np.linalg.norm(cross_mean)

                solution = cross_mean

            else:
                # pick random color from matplotlib tab and save in solution as 1x3 numpy array RGB
                randi = np.random.randint(0, 10)
                solution = np.array(plt.get_cmap('tab10')(randi)[:3])

            cross_mean = solution

            # encode patch points with feature
            # pointcloud_arr[candidate_id, 3:6] = cross_mean
            # pointcloud_enc[patch_idx[candidate_id], 3:6] = cross_mean # legacy proper
            pointcloud_enc[pt_id, 3:6] = cross_mean
            pointcloud_end_rgb = copy.deepcopy(pointcloud_enc)
            pointcloud_end_rgb[:, 3:6] = np.abs(pointcloud_end_rgb[:, 3:6]) * 255

            if plot:
                plot_cloud(pc=patch, head='ok', candidate=candidate, c_grav=c_grav, cross_mean=cross_mean,
                           norms=patch_norms)

            # # v3: gaussian elimintation
            # A = patch_norms
            # augmented_matrix = np.hstack((A.T, np.zeros((A.shape[1], 1))))
            # for i in range(augmented_matrix.shape[0]):
            #     # Find the pivot row and pivot element
            #     pivot_row = i
            #     pivot_element = augmented_matrix[i, i]
            #     for j in range(i, augmented_matrix.shape[0]):
            #         if np.abs(augmented_matrix[j, i]) > np.abs(pivot_element):
            #             pivot_row = j
            #             pivot_element = augmented_matrix[j, i]
            #     # Swap the pivot row with the current row if necessary
            #     if pivot_row != i:
            #         augmented_matrix[[pivot_row, i], :] = augmented_matrix[[i, pivot_row], :]
            #     # Normalize the pivot row
            #     augmented_matrix[i, :] /= pivot_element
            #     # Eliminate the non-zero elements below the pivot
            #     for j in range(i + 1, augmented_matrix.shape[0]):
            #         factor = augmented_matrix[j, i]
            #         augmented_matrix[j, :] -= factor * augmented_matrix[i, :]
            #
            # solution = np.zeros((A.shape[1], 1))
            # for i in range(A.shape[1] - 1, -1, -1):
            #     solution[i] = -np.sum(augmented_matrix[i, i + 1:-1] * solution[i + 1:]) / augmented_matrix[i, i]
            #
            # solution = solution.ravel()
            # solution = solution / np.linalg.norm(solution)
            # print(solution)
            # cross_mean = solution

            # check if even, if no remove last entry
            # if len(patch_norms) % 2 != 0:
            #     patch_norms = patch_norms[:-1, :]
            # # pairwise crossproduct
            # cross = np.cross(patch_norms[0::2, :], patch_norms[1::2, :])
            # # calc mean of all cross products
            # cross_mean = np.mean(cross, axis=0)
            # cross_mean = cross_mean / np.linalg.norm(cross_mean)

            # # v2: gram schmidt
            # # https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
            # tol = 1e-6
            # vectors = patch_norms
            # vectors[0] /= np.linalg.norm(vectors[0])
            # orthogonal_vectors = [vectors[0]]
            # for i in range(len(patch_norms)):
            #     v = vectors[i]
            #     projections = [np.dot(v, u) * u for u in orthogonal_vectors]
            #     orthogonal_vector = v - sum(projections)
            #     if np.linalg.norm(orthogonal_vector) < tol:
            #         orthogonal_vector = np.zeros_like(orthogonal_vector)
            #     else:
            #         orthogonal_vector /= np.linalg.norm(orthogonal_vector)
            #     orthogonal_vectors.append(orthogonal_vector)
            # cross_mean = orthogonal_vectors[-1]

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

    pc_end_rgb = pointcloud_end_rgb[~np.all(pointcloud_end_rgb[:, 3:6] == 0, axis=1)]
    with open('C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/test_beams_ok_enc_rgb.txt', 'w') as f:
        np.savetxt(f, pc_end_rgb, fmt='%.6f')

# #    scatter plot points with iso view
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], s=0.05)
#     ax.set_aspect('equal')
#     plt.show()
#
#     # scatter plot points with iso view
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(ok_cp[:, 0], ok_cp[:, 1], ok_cp[:, 2], s=0.05)
#     ax.set_aspect('equal')
#     plt.show()
