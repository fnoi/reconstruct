import copy

import numpy as np
import open3d as o3d

from sklearn.cluster import DBSCAN


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
    a = 0
    # 1-3 coordinate 4-6 normal 7-9 supernormal 10 confidence 11 ransac/dbscan cluster label
    # initiate first cluster with highest confidence supernormal point
    first_point = np.argmax(points[:, 9])
    ball_radius = config.clustering.dist_thresh_dbscan
    # add all touched dbscan clusters to this cluster
    # neighborhood checks for all points in the cluster
    # grow where supernormal deviation within limits
    # for each iteration add full dbscans if applicable
    # store info to a new label column