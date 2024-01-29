import copy

import numpy as np
import open3d as o3d

from sklearn.cluster import DBSCAN

from instance_segmentation import angle_between_normals
from tools.local import supernormal_svd


def ransac_dbscan_subsequent(array_working, config):
    array_source = copy.deepcopy(array_working)

    min_count = config.clustering.count_thresh_ransac
    point_labels = np.zeros(len(array_source))

    break_count = 0
    ransac_labels = np.zeros(len(array_source))
    ransac_label = 0

    while True:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(array_working[:, :3])
        ransac_plane, ransac_inliers = point_cloud.segment_plane(
            distance_threshold=config.clustering.dist_thresh_ransac,
            ransac_n=5,
            num_iterations=config.clustering.iter_thresh_ransac
        )
        if len(ransac_inliers) > min_count:
            ransac_label += 1
            print(ransac_label)
            # label the inliers
            point_xyz = array_working[ransac_inliers, :3]
            # find id in source points

            for point in point_xyz:
                source_point_id = np.where(np.all(array_source[:, :3] == point, axis=1))[0]
                ransac_labels[source_point_id] = ransac_label
            # remove inliers from working points
            array_working = np.delete(array_working, ransac_inliers, axis=0)


        else:
            min_count -= 1
            if min_count < config.clustering.count_thresh_ransac_rest:
                print('Minimum count threshold reached, stopping RANSAC.')
                break

    print('RANSAC finished, starting DBSCAN.')
    # note 0 is unclustered, unplaned
    dbscan_labels = np.zeros(len(array_source))
    dbscan_label = 0
    # list of ransac labels
    ransac_label_list = [i for i in range(1, int(np.max(ransac_labels)))]
    ransac_label_list.append(0)
    for working_label in ransac_label_list:
        print(f'Working on label {working_label}')
        # active array is where ransac labels are working_label
        array_working = array_source[ransac_labels == working_label]
        dbscan_clustering = DBSCAN(
            eps=config.clustering.dist_thresh_dbscan,
            min_samples=config.clustering.count_thresh_dbscan
        ).fit(array_working[:, :3])
        dbscan_cluster_labels = dbscan_clustering.labels_
        # check if all labels are -1
        if np.all(dbscan_cluster_labels == -1):
            if working_label == 0:
                continue
            else:
                ransac_labels[ransac_labels == working_label] = working_label + 1

        else:

            for i in range(np.max(dbscan_cluster_labels) + 1):
                dbscan_label += 1
                for j in range(len(dbscan_cluster_labels)):
                    if dbscan_cluster_labels[j] == i:
                        # per point find xyz
                        point_xyz = array_working[j, :3]
                        # find id in source points
                        source_point_id = np.where(np.all(array_source[:, :3] == point_xyz, axis=1))[0]
                        dbscan_labels[source_point_id] = dbscan_label




    return dbscan_labels





def region_growing_ransac(array_working, config):
    array_source = copy.deepcopy(array_working)

    min_count = config.clustering.count_thresh_ransac
    point_labels = np.zeros(len(array_source))

    ransac_iteration_count = 0

    while True:

        point_cloud_current = o3d.geometry.PointCloud()
        point_cloud_current.points = o3d.utility.Vector3dVector(array_working[:, :3])
        ransac_plane, ransac_inliers = point_cloud_current.segment_plane(
            distance_threshold=config.clustering.dist_thresh_ransac,
            ransac_n=5,
            num_iterations=config.clustering.iter_thresh_ransac
        )


        if len(ransac_inliers) > min_count:
            dbscan_fail_count = 0
            print(np.max(point_labels))
            while True:
                print('entering dbscan')
                point_labels, array_working, dbscan_fail_count = dbscan_clustering_on_plane(
                    working_points=array_working,
                    plane_ids=ransac_inliers,
                    source_points=array_source,
                    source_labels=point_labels,
                    config=config,
                    dbscan_fail_count=dbscan_fail_count
                )
                print(dbscan_fail_count)
                if dbscan_fail_count == 0 or dbscan_fail_count > 10:
                    break

        else:
            min_count -= 1

            if min_count < config.clustering.count_thresh_ransac_rest:
                print('Minimum count threshold reached, stopping RANSAC.')

                break

        ransac_iteration_count += 1

        if ransac_iteration_count > config.clustering.max_ransac_iterations:
            print('Maximum RANSAC iterations reached, stopping RANSAC.')

            break

    return point_labels

def dbscan_clustering_on_plane(working_points, plane_ids,
                               source_points, source_labels,
                               config, dbscan_fail_count):
    dbscan_mask = np.ones(len(working_points), dtype=bool)
    plane_points = working_points[plane_ids]
    dbscan_clustering = DBSCAN(
        eps=config.clustering.dist_thresh_dbscan,
        min_samples=config.clustering.count_thresh_dbscan
    ).fit(plane_points[:, :3])
    dbscan_cluster_labels = dbscan_clustering.labels_
    # check if all labels are -1
    if np.all(dbscan_cluster_labels == -1):  # no valid clusters
        return source_labels, working_points, dbscan_fail_count + 1
    else:
        current_label = int(np.max(source_labels))
        for i in range(np.max(dbscan_cluster_labels) + 1):
            current_label += 1

            for j in range(len(dbscan_cluster_labels)):
                if dbscan_cluster_labels[j] == i:
                    # per point find xyz
                    point_xyz = plane_points[j, :3]
                    # find id in source points
                    source_point_id = np.where(np.all(source_points[:, :3] == point_xyz, axis=1))[0]
                    source_labels[source_point_id] = current_label
                    dbscan_mask[j] = False

        # remove points from working points based on mask
        working_points = working_points[dbscan_mask]

        return source_labels, working_points, dbscan_fail_count

def region_growing_ransac_dbscan_supernormals(points, config):
    # 1-3 coordinate 4-6 normal 7-9 supernormal
    # 10 confidence 11 ransac/dbscan cluster label 12 region growing cluster label
    # initiate first cluster with highest confidence supernormal point
    cluster_labels = np.zeros(len(points))
    points_array = np.concatenate((points, cluster_labels.reshape(-1, 1)), axis=1)
    points_cloud = o3d.geometry.PointCloud()
    points_cloud.points = o3d.utility.Vector3dVector(points_array[:, :3])
    points_cloud.normals = o3d.utility.Vector3dVector(points_array[:, 3:6])
    points_tree = o3d.geometry.KDTreeFlann(points_cloud)
    ball_radius = config.clustering.dist_thresh_ball

    label = 0
    iter = 0

    while True:
        # temporary array with only unlabelled points
        _t = points_array[points_array[:, 11] == 0]
        if len(_t) == 0:  # all points have been clustered
            break
        else:
            label += 1
            # identify seed point by confidence (highest-rated supernormal in unlabelled points)
            # temporary array with only dbscan labelled points
            _t = _t[_t[:, 10] != 0]
            # find id of point with lowest confidence
            _i = np.argmin(_t[:, 9])
            _s = _t[_i, :]

            # find index of _s in points_array
            seed = np.where((points_array == _s).all(axis=1))[0][0]

            cluster_labels[seed] = label
            points_array[seed, 11] = label
            cluster_ptx = [seed]
            visited_ptx = [seed]
            print(f'seed {seed}, label {label}')
            seed_id = 0

            cluster_supernormal = points_array[seed, 6:9]

            while True:
                if seed_id > len(cluster_ptx):
                    break
                else:
                    seed = cluster_ptx[seed_id]
                    # 1. find all potential candidates
                    candidates = points_tree.search_radius_vector_3d(points_array[seed, :3], ball_radius)[1]
                    candidates = list(np.asarray(candidates).flatten())
                    # remove points that are already clustered
                    valid_candidates = []
                    for candidate in candidates:
                        if candidate not in cluster_ptx and points_array[candidate, 11] == 0 and candidate not in visited_ptx:
                            valid_candidates.append(candidate)
                    print(f'seed id {seed_id}, seed {seed}, cluster size {len(cluster_ptx)}, {len(valid_candidates)} valid candidates, length visited {len(visited_ptx)}')
                    # if cache is empty, cluster growth stops.
                    if len(valid_candidates) == 0:
                        break

                    # cluster_normals = points_array[cluster_ptx, 3:6]
                    # cluster_supernormal = supernormal_svd(cluster_normals)

                    # 2. evaluate the narrowed down candidates, extend cluster, track cache
                    for candidate in valid_candidates:
                        # calculate supernormal deviation
                        # if within limits add to cluster
                        # calculate cluster-supernormal angle
                        # cluster_normals = points_array[cluster_ptx, 3:6]
                        # cluster_supernormal = supernormal_svd(cluster_normals)

                        angular_diff = angle_between_normals(cluster_supernormal, points_array[candidate, 3:6])
                        # print(np.rad2deg(angular_diff))
                        angular_diff = abs(np.rad2deg(angular_diff) - 90)
                        # print(angular_diff)

                        if angular_diff < 5:

                        # if angle_between_normals(points_array[seed, 6:9], points_array[candidate, 6:9]) < config.clustering.angle_thresh_supernormal:
                            # find all points with same ransac/dbscan cluster label
                            if points_array[candidate, 10] == 0:  # no cluster from ransac/dbscan: only 1 candidate
                                ids = [candidate]
                            else:  # cluster from ransac/dbscan: all points from same cluster
                                ids = np.where(points_array[:, 10] == points_array[candidate, 10])[0]
                                # remove ids are already in cluster_ptx
                                ids = [id for id in ids if id not in cluster_ptx]
                                # remove ids that are already clustered or labeled/ only unlabelled points label = 0
                                ids = [id for id in ids if points_array[id, 11] == 0]


                            # ids = np.where(points_array[:, 10] == points_array[candidate, 10])[0]
                            # remove ids that are already clustered or labeled

                            cluster_ptx.extend(ids)
                            cluster_labels[ids] = label
                            points_array[ids, 11] = label

                            # print(f'cluster size: {len(cluster_ptx)}, of {len(points_array)}')
                            # cluster_ptx.append(candidate)  # append all cluster points instead of just the seed
                            visited_ptx = np.unique(visited_ptx + ids).tolist()


                        else:
                            visited_ptx = np.unique(visited_ptx + ids).tolist()


                    seed_id += 1

            # assign the label to all points in the cluster
            # cluster_labels[cluster_ptx] = label
            # points_array[cluster_ptx, 11] = label
            # label += 1

    return cluster_labels