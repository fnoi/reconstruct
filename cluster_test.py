from math import floor
import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def euclidian_distancex(p1x, p2x):
    p1x = np.asarray(p1x)
    p2x = np.asarray(p2x)

    if p1x.shape != p2x.shape:
        raise ValueError('p1x and p2x must have the same shape')

    distance = np.sqrt(np.sum((p1x - p2x) ** 2, axis=1))

    return distance



def angular_distance(v1, v2):
    v1_magnitude = np.linalg.norm(v1)
    v2_magnitude = np.linalg.norm(v2)
    if v1_magnitude == 0 or v2_magnitude == 0:
        raise ValueError('Cannot compute angle between zero vectors')
    dot_product = np.dot(v1, v2)
    cos_angle = dot_product / (v1_magnitude * v2_magnitude)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    angle = np.degrees(angle)

    return angle


def angular_distancex(v1x, v2x):
    v1x = np.asarray(v1x)
    v2x = np.asarray(v2x)

    if v1x.shape[1:] != (3,) or v2x.shape[1:] != (3,):
        raise ValueError('v1x and v2x must have the shape (n, 3)')

    v1_magnitude = np.linalg.norm(v1x, axis=1)
    v2_magnitude = np.linalg.norm(v2x, axis=1)

    zero_mask = (v1_magnitude == 0) | (v2_magnitude == 0)
    if zero_mask.any():
        raise ValueError('Cannot compute angle between zero vectors')

    dot_product = np.sum(v1x * v2x, axis=1)
    cos_anglex = dot_product / (v1_magnitude * v2_magnitude)
    cos_anglex = np.clip(cos_anglex, -1, 1)
    anglex = np.arccos(cos_anglex)
    anglex = np.degrees(anglex)

    return anglex


def region_growing(points, spatial_threshold, feature_threshold):
    visited_points = np.zeros(len(points), dtype=bool)
    clusters = []

    kdtree = KDTree(points[:, :3], leaf_size=2)
    pt_idx = np.random.permutation(np.arange(len(points)))

    for pt_id, point in zip(pt_idx, points):
        if not visited_points[pt_id]:
            cluster = grow_cluster(pt_id, points, visited_points, spatial_threshold, feature_threshold, kdtree)
            clusters.append(cluster)
            print(f'{pt_id} - {len(cluster)} - {len(points) - np.count_nonzero(visited_points)}')

    return clusters


def grow_cluster(seed_idx, points, visited_points, spatial_threshold, feature_threshold, kdtree):
    if type(seed_idx) == np.int32:  # initialize region
    # if len(seed_idx) == 1:  # initialize region
        cluster = [seed_idx]
        visited_points[seed_idx] = True
        neighbors = kdtree.query_radius([points[seed_idx][:3]], r=spatial_threshold)[0]
        neighbor_fts = points[neighbors][:,3:]
        neighbor_fts_dist = angular_distancex(np.tile(points[seed_idx][3:], (neighbor_fts.shape[0], 1)), neighbor_fts)
        a = 0

    neighbors = kdtree.query_radius([points[seed_idx][:3]], r=spatial_threshold)[0]
    neighbor_fts = points[neighbors][:,3:]
    neighbor_fts_dist = [angular_distance(points[seed])]
    valid_idx = np.where(angular_distance(points[seed_idx][3:], neighbor_fts) <= feature_threshold)[0]




    i = 0

    while i < len(cluster):
        current_point = points[cluster[i]]
        spatial_neighbors = kdtree.query_radius([current_point[:3]], r=spatial_threshold)[0]

        for neighbor_idx in spatial_neighbors:
            if not visited_points[neighbor_idx]:
                candidate_point = points[neighbor_idx]
                feature_distance = angular_distance(current_point[3:], candidate_point[3:])

                if feature_distance <= feature_threshold:
                    cluster.append(neighbor_idx)
                    visited_points[neighbor_idx] = True
        i += 1

    return cluster


if __name__ == "__main__":

    with open('C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/test_beams_ok_enc.txt', 'r') as f:
        data = np.loadtxt(f)
    # with open('C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/test_beams_ok_enc_rgb.txt', 'r') as f:
    #     thisthat = np.loadtxt(f)

    # downsample every *th point
    sampling_rate = 1
    thisthat = data[::sampling_rate, :]
    coordinates = thisthat[:, :3]
    features = thisthat[:, 3:]
    scaler = StandardScaler()
    scaled_feat = scaler.fit_transform(features)
    data = np.hstack((coordinates, scaled_feat))

    search_res = 20
    eps_values = np.linspace(0.1, 2, search_res)

    sv_min, sv_max = 2, 500
    min_samples_values = [_ for _ in range(sv_min, sv_max, int(floor((sv_max - sv_min) / search_res)))]

    best_eps = None
    best_min_samples = None
    best_silhouette = -1

    result_frame = np.zeros((len(eps_values), len(min_samples_values)))

    finder = False
    if finder:

        for i, eps in enumerate(eps_values):
            for j, min_samples in enumerate(min_samples_values):
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(data)
                if len(np.unique(cluster_labels)) == 1:
                    silhouette = -1
                else:
                    silhouette = silhouette_score(data, cluster_labels)
                print(eps, min_samples, silhouette)
                result_frame[i, j] = silhouette
                if silhouette > best_silhouette:
                    best_eps = eps
                    best_min_samples = min_samples
                    best_silhouette = silhouette

        print(f'Best eps: {best_eps}')
        print(f'Best min_samples: {best_min_samples}')
        print(f'Best silhouette: {best_silhouette}')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(result_frame, cmap='rainbow')
        plt.show()

    else:

        # dbscan = DBSCAN(eps=0.25, min_samples=100)
        #
        # cluster_labels = dbscan.fit_predict(data)
        spatial_threshold = 0.1
        feature_threshold = 15

        clusters = region_growing(data, spatial_threshold, feature_threshold)

        cluster_labels = np.zeros(len(data))
        i = 0
        for cluster in tqdm(clusters, desc='assigning labels'):
            cluster_labels[cluster] = i
            i += 1

        # describe clusters
        cluster_sizes = np.zeros(len(clusters))
        for i, cluster in enumerate(clusters):
            cluster_sizes[i] = len(cluster)
        print(f'Cluster sizes: {cluster_sizes}')
        print(f'Number of clusters: {len(clusters)}')


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=cluster_labels, cmap='rainbow', s=0.1)
        plt.show()

        # append cluster labels to data
        data = np.hstack((data, cluster_labels.reshape(-1, 1)))

    # save to txt
    with open('C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/test_beams_clusters.txt', 'w') as f:
        np.savetxt(f, data, fmt='%i')

    # encode cluster_labels in RGB

    # plot heatmap of results

    a = 0
