import copy
import pathlib
import os
import pickle
import random

import dev_outsource

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import plotly.graph_objs as go
from omegaconf import OmegaConf
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from tqdm import tqdm
from sklearn.cluster import DBSCAN

import tools.geometry
from tools.local import supernormal_svd_s1, supernormal_confidence, angular_deviation
from tools.utils import find_random_id


def get_neighbors(point_id, full_cloud, dist):
    coords_point_check = cloud_c_n_sn_co[point_id, :3]
    coords_points_all = cloud_c_n_sn_co[:, :3]
    distances = cdist(coords_point_check[None, :], coords_points_all)
    ids_neighbors = np.where(distances < dist)[1]
    return ids_neighbors.tolist()


def get_neighbors_flex(point_ids, full_cloud, dist):
    coords_points_check = cloud_c_n_sn_co[point_ids, :3]
    coords_points_all = cloud_c_n_sn_co[:, :3]
    distances = cdist(coords_points_check, coords_points_all)
    ids_neighbors = np.where(distances < dist)[1]
    # delete seed points
    ids_neighbors = np.delete(ids_neighbors, point_ids)
    return ids_neighbors.tolist()


def angle_between_normals(normal1, normal2):
    """Calculate the angle between two normals."""
    dot_product = np.dot(normal1, normal2)
    return np.arccos(dot_product / (np.linalg.norm(normal1) * np.linalg.norm(normal2)))


def region_growing_ransac(points_array, config):
    array_true = copy.deepcopy(points_array)

    label = 1
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_array[:, :3])

    min_count = config.clustering.count_thresh_ransac
    point_labels = np.zeros(len(points_array))
    unsuccessful_dbscan_count = 0
    ransac_count = 0

    while True:

        # print(f'running ransac with min_count {min_count}')
        iter = 0
        min_count -= 1
        while True:
            iter += 1
            # print(f'iteration {iter}')
            if len(point_cloud.points) < config.clustering.count_thresh_ransac_rest:
                break
            plane_model, inliers = point_cloud.segment_plane(distance_threshold=config.clustering.dist_thresh_ransac,
                                                             ransac_n=5,
                                                             num_iterations=config.clustering.iter_thresh_ransac)
            if len(inliers) > min_count:  # current (decreasing) min_count
            # if len(inliers) > config.clustering.count_thresh_ransac:
                ransac_count += 1
                print(f'RANSAC cluster {ransac_count} with {len(inliers)} points found in point cloud w {len(point_cloud.points)} points')

                inlier_points_xyz = points_array[inliers, :3]  # get xyz of inlier points for dbscan clustering
                clustering = DBSCAN(eps=config.clustering.dist_thresh_dbscan,
                                    min_samples=config.clustering.count_thresh_dbscan).fit(inlier_points_xyz)
                clustering_labels = clustering.labels_
                unique_labels = np.unique(clustering_labels)

                if len(unique_labels) > 1:
                    # drop noise-labelled points from labels and inliers
                    unique_labels = unique_labels[unique_labels != -1]
                    print(f'{len(unique_labels)} clusters found with DBSCAN')

                    sucessfully_clustered_ids = np.where(clustering_labels != -1)[0]

                    for cluster_label in unique_labels:
                        cluster_label_ids = np.where(clustering_labels == cluster_label)[0]

                        for cluster_labeled_point in cluster_label_ids:
                            true_id = np.where(np.all(array_true[:, :3] == inlier_points_xyz[cluster_labeled_point], axis=1))[0][0]
                            point_labels[true_id] = label
                            # write acutal label to cluster
                            # cluster.append(inlier_id)
                            # write label to ids

                        label += 1
                    # remove sucessfully clustered points from point cloud
                    mask = np.ones(len(point_cloud.points), dtype=bool)
                    mask[sucessfully_clustered_ids] = False
                    points_array = points_array[mask]
                    # remaining_xyz = np.array(point_cloud.points)[mask]
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(points_array[:, :3])


                else:
                    unsuccessful_dbscan_count += 1
            else:

                break

        if min_count < config.clustering.count_thresh_ransac_rest or unsuccessful_dbscan_count > 10:
            break

    # find idxed points in original numpy cloud
    return point_labels



def region_growing_with_kdtree(points, feature_cloud_tree, config):
    points = np.array(points[:, :6])

    distance_threshold = config.clustering.dist_thresh_normal
    angle_threshold = config.clustering.angle_thresh_normal
    angle_threshold = np.deg2rad(angle_threshold)

    num_points = points.shape[0]
    clusters = [-1] * num_points  # Initialize cluster labels to -1
    cluster_id = 0
    iter = 0
    for i in range(num_points):
        if clusters[i] == -1:  # Unprocessed point
            clusters[i] = cluster_id
            seeds = [i]

            while seeds:
                current_index = seeds.pop()
                iter += 1
                print(f'current_index {current_index}, iteration {iter}')
                current_point = points[current_index]

                # Query points within the distance threshold
                indices = feature_cloud_tree.query_ball_point(current_point[:3], distance_threshold)

                for j in indices:
                    if clusters[j] == -1:  # Unprocessed point
                        # calculate mean normal of cluster
                        regional_normal = np.mean(points[np.where(np.array(clusters) == cluster_id)][:, 3:], axis=0)
                        regional_normal /= np.linalg.norm(regional_normal)

                        if angle_between_normals(regional_normal, points[j, 3:]) < angle_threshold:
                            clusters[j] = cluster_id
                            seeds.append(j)
            print(f'\n\ncluster {cluster_id} done, cluster size {clusters.count(cluster_id)}\n\n')
            cluster_id += 1

    return clusters


def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(point1 - point2)


def angle_between_supernormals(supernormal1, supernormal2):
    """Calculate the angle between two supernormals."""
    dot_product = np.dot(supernormal1, supernormal2)
    return np.arccos(dot_product / (np.linalg.norm(supernormal1) * np.linalg.norm(supernormal2)))


def region_growing_supernormals(points, kdtree, config):
    distance_threshold = config.clustering.max_dist_euc
    angle_threshold = config.clustering.angle_thresh_supernormal
    angle_threshold = np.deg2rad(angle_threshold)

    num_points = points.shape[0]
    super_clusters = [-1] * num_points  # Initialize super cluster labels to -1
    super_cluster_id = 0

    # get list of normal cluster ids sorted by cluster size
    cluster_ids, cluster_sizes = np.unique(points[:, -2], return_counts=True)
    cluster_ids_sorted = cluster_ids[np.argsort(cluster_sizes)]

    # get list of id args sorted by confidence, minimum first
    confidence_sorted_ids = np.argsort(points[:, -1])
    confidence_sorted_ids = confidence_sorted_ids.tolist()

    # sort confidence sorted ids by normal cluster size
    confidence_sorted_ids = [x for x in confidence_sorted_ids if points[x, -2] in cluster_ids_sorted]

    for i in confidence_sorted_ids:
        if super_clusters[i] == -1:  # Unprocessed point
            initial_cluster = points[i, -1]  # Last column represents the initial cluster label
            super_clusters[i] = super_cluster_id

            # all points of same cluster get super cluster id
            seed_ids = np.where(points[:, -1] == initial_cluster)[0].tolist()
            # write super cluster id to all points of same cluster
            super_clusters_array = np.array(super_clusters)
            super_clusters_array[seed_ids] = super_cluster_id
            super_clusters = super_clusters_array.tolist()
            # overwrite supernormals in cluster with supernormal of seed point
            points[seed_ids, 6:9] = points[i, 6:9]

            seeds = [i]

            while seeds:
                current_index = seeds.pop()
                current_point = points[current_index]

                # Query points within the distance threshold
                indices = kdtree.query_ball_point(current_point[:3], distance_threshold)

                for j in indices:
                    if super_clusters[j] == -1:  # Unprocessed point
                        if angle_between_supernormals(current_point[6:9], points[j, 6:9]) < angle_threshold:
                            # add this point to the cluster
                            super_clusters[j] = super_cluster_id
                            seeds.append(j)
                            # add all points of same normal cluster to super cluster
                            seed_ids = np.where(points[:, -1] == points[j, -1])[0].tolist()
                            super_clusters_array = np.array(super_clusters)
                            super_clusters_array[seed_ids] = super_cluster_id
                            super_clusters = super_clusters_array.tolist()
                            # overwrite supernormals in cluster with supernormal of seed point
                            points[seed_ids, 6:9] = points[i, 6:9]

            super_cluster_id += 1

    return super_clusters


def region_growth_rev(feature_cloud_c_n_sn_co, config, feature_cloud_tree):
    regional_labels = np.zeros(len(feature_cloud_c_n_sn_co))
    regional_label = 0
    # append labels to cloud
    cloud_c_n_sn_co_l = np.concatenate((feature_cloud_c_n_sn_co, regional_labels[:, None]), axis=1)

    while True:
        regional_label += 1
        # find all points with no label
        no_label_row_ids = cloud_c_n_sn_co_l[:, -1] == 0
        # find no-label point with lowest co value
        region_seed = np.argmin(cloud_c_n_sn_co_l[no_label_row_ids, -2])
        region_neighbors = []
        region_points = [region_seed]
        individual_neighbors = get_neighbors(point_id=region_seed,
                                             full_cloud=cloud_c_n_sn_co_l,
                                             dist=config.clustering.max_dist_euc)
        region_neighbors.extend(individual_neighbors)
        region_neighbors = list(np.unique(region_neighbors))

        # check supernormal dev between seed and neighbors
        for neighbor in individual_neighbors:
            supernormal_deviation = angular_deviation(
                vector=cloud_c_n_sn_co_l[neighbor, 6:9],
                reference=cloud_c_n_sn_co_l[region_seed, 6:9]
            )
            if supernormal_deviation < config.clustering.angle_thresh_supernormal:
                cloud_c_n_sn_co_l[neighbor, -1] = regional_label
                region_points.append(neighbor)

        # visiting_region_point_ind = 0
        legacy_neighbors = copy.deepcopy(region_neighbors)
        iter = 0
        checked = [region_seed]
        while True:
            # list of pts to check: not in checked, but in region_points
            check_points = [x for x in region_points if x not in checked]
            if len(check_points) == 0:
                break
            checking_point = check_points[0]
            checked.append(checking_point)
            iter += 1
            if iter > len(region_points):
                break
            print(f'running loop {iter}: point {checking_point}')
            # visiting_region_point_ind += 1
            individual_neighbors = get_neighbors(
                point_id=checking_point,
                full_cloud=cloud_c_n_sn_co_l,
                dist=config.clustering.max_dist_euc
            )
            region_neighbors.extend(individual_neighbors)
            region_neighbors = list(np.unique(region_neighbors))
            # if region_neighbors == legacy_neighbors:
            #     break

            # check supernormal dev between seed and neighbors
            for neighbor in individual_neighbors:
                supernormal_deviation = angular_deviation(
                    vector=cloud_c_n_sn_co_l[neighbor, 6:9],
                    reference=cloud_c_n_sn_co_l[region_seed, 6:9]
                )
                if supernormal_deviation < config.clustering.angle_thresh_supernormal:
                    cloud_c_n_sn_co_l[neighbor, -1] = regional_label
                    region_points.append(neighbor)

            legacy_neighbors = copy.deepcopy(region_neighbors)

        # save cloud
        with open(f'{basepath}{config.general.project_path}data/parking/region_{regional_label}.txt', 'w') as f:
            np.savetxt(f, cloud_c_n_sn_co_l, fmt='%.6f', delimiter=';', newline='\n')
        # make o3d point cloud
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_c_n_sn_co_l[:, :3])
        # add l as scalar
        colormap = plt.get_cmap("tab20")
        colors = colormap(cloud_c_n_sn_co_l[:, -1])
        cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # store point cloud to pcd file
        # write normals from 6:9
        cloud.normals = o3d.utility.Vector3dVector(cloud_c_n_sn_co_l[:, 6:9])
        o3d.io.write_point_cloud(f'{basepath}{config.general.project_path}data/parking/region_{regional_label}.pcd',
                                 cloud)
        a = 0
        # check 2

        # write labels

        # be done (with first region)

    cloud_c_n_sn_co_l[lowest_co, -1] = regional_label

    a = 0


def region_growth(feature_cloud_c_n_sn_co, config, feature_cloud_tree):
    points_region = []
    # start with high confidence (low value)
    start_id = np.argmin(cloud_c_n_sn_co[:, -1])
    points_active = [start_id]

    for point_check in points_active:
        print(f'point_check {point_check}')
        check_idx = get_neighbors(point_id=point_check,
                                  full_cloud=cloud_c_n_sn_co,
                                  dist=config.clustering.max_dist_euc)
        idx_append = np.zeros(len(check_idx))
        for i, check_id in enumerate(check_idx):
            a = 0
            # print(f'check_id {check_id}')
            supernormal_deviation = angular_deviation(cloud_c_n_sn_co[check_id, 6:9], cloud_c_n_sn_co[point_check, 6:9])
            if supernormal_deviation < config.clustering.angle_thresh_supernormal:
                # points_region.append(check_id)
                idx_append[i] = 1
        # ids of idx_append that are 1
        idx_append = np.where(idx_append == 1)[0]
        append_actual = [check_idx[_] for _ in idx_append]
        points_region.append(append_actual)
        # remove appended from check_idx
        check_idx = np.delete(check_idx, idx_append)
        if len(check_idx) > 0 and len(points_region) > 0:
            for check_id in check_idx:
                # check distance to all points in the region
                dists = []
                for point_region in points_region:
                    dist = np.linalg.norm(cloud_c_n_sn_co[check_id, :3] - cloud_c_n_sn_co[point_region, :3])
                    dists.append(dist)
                # sort distances and ids
                dists = np.array(dists)
                ids = np.argsort(dists)
            for id in ids:
                # check if closest point is within threshold
                if dists[id] < config.clustering.angle_thresh_normal:
                    points_region.append(id)
                a = 0
    b = 0

    return None


def calc_supernormals(point_coords_arr, point_normals_arr, point_ids_all, point_cloud_tree):
    supernormals = []
    confidences = []

    plot_id = random.randint(0, len(point_ids_all))

    for id_seed in tqdm(point_ids_all, desc='computing super normals', total=len(point_ids_all)):
        point_seed = point_coords_arr[id_seed]

        ids_neighbors = point_cloud_tree.query_ball_point(point_seed, r=config.local.supernormal_radius)
        normals_neighbors = point_normals_arr[ids_neighbors]
        # normalize normals
        normals_neighbors /= np.linalg.norm(normals_neighbors, axis=1)[:, None] * 5
        normal_seed = point_normals_arr[id_seed] / np.linalg.norm(point_normals_arr[id_seed])
        normals_patch = np.concatenate((normals_neighbors, normal_seed[None, :]), axis=0)

        coords_neighbors = point_coords_arr[ids_neighbors]
        arrowheads = coords_neighbors + normals_neighbors

        supernormal, sig_1, sig_2, sig_3 = supernormal_svd_s1(normals_patch, full_return=True)
        confidence = supernormal_confidence(supernormal, normals_patch, sig_1, sig_3)
        confidences.append(confidence)

        supernormals.append(supernormal)
        supernormal /= np.linalg.norm(supernormal) * 5

        plot_flag = False
        plot_id = 3781
        if id_seed == plot_id:
            plot_flag = True
        # override
        # plot_flag = False
        if plot_flag:
            fig = go.Figure()

            # Add scatter plot for the points
            fig.add_trace(go.Scatter3d(
                x=point_coords_arr[ids_neighbors, 0],
                y=point_coords_arr[ids_neighbors, 1],
                z=point_coords_arr[ids_neighbors, 2],
                mode='markers',
                # marker type +
                marker=dict(
                    size=3,
                    color='black',
                    opacity=0.6  # ,
                    # symbol='dot'
                )
            ))

            # Add lines for the normals
            for i in range(len(coords_neighbors)):
                # Start point of each line (origin)
                x0, y0, z0 = coords_neighbors[i, :]
                # End point of each line (origin + normal)
                x1, y1, z1 = coords_neighbors[i, :] + normals_neighbors[i, :]

                fig.add_trace(go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode='lines',  # Set to lines mode
                    line=dict(color='blue', width=0.5)  # Adjust line color and width as needed
                ))

            # add red thick line for supernormal from seed coord
            x0, y0, z0 = point_seed
            x1, y1, z1 = point_seed + supernormal
            fig.add_trace(go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode='lines',  # Set to lines mode
                line=dict(color='green', width=10)  # Adjust line color and width as needed
            ))
            # headline
            fig.update_layout(
                title_text=f'Point {id_seed} with {len(ids_neighbors)} neighbors, confidence {confidence:.2f}',
                scene=dict(
                    xaxis=dict(showbackground=False, showgrid=False, zeroline=False),
                    yaxis=dict(showbackground=False, showgrid=False, zeroline=False),
                    zaxis=dict(showbackground=False, showgrid=False, zeroline=False),
                    bgcolor='white'  # Change plot background to white (or any other color)
                )
            )
            fig.update_layout()

            # Show the figure
            fig.show()

    # supernormal abs for each axis
    supernormals = np.abs(np.array(supernormals))

    return supernormals, confidences


if __name__ == "__main__":
    config = OmegaConf.load('config.yaml')
    # local runs only, add docker support
    if os.name == 'nt':
        basepath = config.general.basepath_windows
    else:  # os.name == 'posix':
        basepath = config.general.basepath_macos
    path = pathlib.Path(f'{basepath}{config.general.project_path}{config.segmentation.cloud_path}')

    compute = True  # TODO switch back once we have nice confidenzia
    if compute:

        # read point cloud from file to numpy array
        with open(path, 'r') as f:
            point_coords_arr = np.loadtxt(f)[:, :3]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_coords_arr)
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.local.normals_radius, max_nn=config.local.max_nn))

        point_normals_arr = np.asarray(point_cloud.normals)
        # store point cloud to pcd file
        o3d.io.write_point_cloud(f'{basepath}{config.general.project_path}data/parking/normals.ply', point_cloud)

        point_ids_all = [i for i in range(point_coords_arr.shape[0])]
        point_ids_done = []
        point_cloud_tree = KDTree(point_coords_arr)

        supernormals, confidences = calc_supernormals(point_coords_arr, point_normals_arr, point_ids_all, point_cloud_tree)

        # concat with original point cloud
        cloud_c_sn = np.concatenate((point_coords_arr, supernormals), axis=1)

        cloud_c_sn_co = np.concatenate((cloud_c_sn, np.array(confidences)[:, None]), axis=1)
        with open(f'{basepath}{config.general.project_path}data/parking/handover_supernormals_confidences2.txt', 'w') as f:
            np.savetxt(f, cloud_c_sn_co, fmt='%.6f')


        cloud_c_n_sn_co = np.concatenate((point_coords_arr, point_normals_arr, supernormals), axis=1)
        cloud_c_n_sn_co = np.concatenate((cloud_c_n_sn_co, np.array(confidences)[:, None]), axis=1)

        point_labels = dev_outsource.ransac_dbscan_subsequent(cloud_c_n_sn_co, config)

        full_cloud = np.concatenate((cloud_c_n_sn_co, np.array(point_labels)[:, None]), axis=1)
        # store full cloud as numpy array
        with open(f'{basepath}{config.general.project_path}data/parking/full_cloud.npy', 'wb') as f:
            np.save(f, full_cloud)
    else:
        with open(f'{basepath}{config.general.project_path}data/parking/full_cloud.npy', 'rb') as f:
            full_cloud = np.load(f)

    paper_flag = False
    if paper_flag:
        full_cloud, dirt_cloud = dev_outsource.region_growing_ransac_dbscan_supernormals(full_cloud, config)
        # delete all columns except xyz and labels

    else:
        with open(f'{basepath}{config.general.project_path}data/in_test/test_junction_segmentation_results.txt', 'r') as f:
            full_cloud = np.loadtxt(f, delimiter=' ') # , usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
        cloud_frame = pd.DataFrame(full_cloud)
        cloud_frame.columns = ['x', 'y', 'z', 'c', 'pp', 'pi', 'gj']

        # find unique ids in pi
        unique_ids = cloud_frame['pi'].unique()
        dirs = {}
        for pi in list(unique_ids):
            if pi == 0:
                continue
            else:
                segment_array = cloud_frame[cloud_frame['pi'] == pi].to_numpy()
                dirs[str(pi)] = tools.geometry.orientation_estimation_s2(segment_array)


        # store the dir dict using pickle
        with open(f'{basepath}{config.general.project_path}data/in_test/dirs.pkl', 'wb') as f:
            pickle.dump(dirs, f, pickle.HIGHEST_PROTOCOL)

        a = 0



    save_cloud = full_cloud[:, [0, 1, 2, -3, -2, -1]]

    # new_cluster_cloud = np.concatenate((full_cloud[:, :3], np.array(cluster_labels)[:, None]), axis=1)
    # write to txt readable by cloudcompare
    with open(f'{basepath}{config.general.project_path}data/parking/new_clustered.txt', 'w') as f:
        np.savetxt(f, save_cloud, fmt='%.6f', delimiter=';', newline='\n')

    with open(f'{basepath}{config.general.project_path}data/parking/dirt_cloud.txt', 'w') as f:
        np.savetxt(f, dirt_cloud, fmt='%.6f', delimiter=';', newline='\n')

    raise Exception('stop here')
    cloud_cluster_ids = region_growing_with_kdtree(cloud_c_n_sn_co, point_cloud_tree, config)
    print(f'found {np.max(cloud_cluster_ids)} clusters')



    # align xyz with labels
    cloud_clustered = np.concatenate((cloud_c_n_sn_co, np.array(cloud_cluster_ids)[:, None]), axis=1)
    # write to txt readable by cloudcompare
    with open(f'{basepath}{config.general.project_path}data/parking/clustered.txt', 'w') as f:
        np.savetxt(f, cloud_clustered, fmt='%.6f', delimiter=';', newline='\n')

    super_clusters = region_growing_supernormals(cloud_clustered, point_cloud_tree, config)
    print(f'found {np.max(super_clusters)} super clusters')

    cloud_super_clustered = np.concatenate((cloud_clustered, np.array(super_clusters)[:, None]), axis=1)

    # DROPPING POINTS
    thresh = 20
    # drop all points that belong to super cluster size lower than 20
    super_cluster_sizes = np.unique(super_clusters, return_counts=True)[1]
    super_cluster_ids = np.unique(super_clusters, return_counts=True)[0]
    super_cluster_ids = super_cluster_ids[super_cluster_sizes > thresh]
    # get ids of points that belong to super clusters with size > thresh
    super_cluster_point_ids = []
    for super_cluster_id in super_cluster_ids:
        super_cluster_point_ids.extend(np.where(super_clusters == super_cluster_id)[0].tolist())
    # drop all points that are not in super_cluster_point_ids
    cloud_super_clustered = cloud_super_clustered[super_cluster_point_ids, :]

    # write to txt readable by cloudcompare
    with open(f'{basepath}{config.general.project_path}data/parking/super_clustered.txt', 'w') as f:
        np.savetxt(f, cloud_super_clustered, fmt='%.6f', delimiter=';', newline='\n')

    # cloud_clustered = region_growth_rev(cloud_c_n_sn_co, config, point_cloud_tree)

    a = 0
