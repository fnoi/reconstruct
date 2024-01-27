import copy

import numpy as np
import open3d as o3d

from sklearn.cluster import DBSCAN


def region_growing_ransac(array_working, config):
    array_source = copy.deepcopy(array_working)

    min_count = config.clustering.count_thresh_ransac
    point_labels = np.zeros(len(array_source))
    dbscan_fail_count = 0

    while True:

        point_cloud_current = o3d.geometry.PointCloud()
        point_cloud_current.points = o3d.utility.Vector3dVector(array_working)
        ransac_plane, ransac_inliers = point_cloud_current.segment_plane(
            distance_threshold=config.clustering.dist_thresh_ransac,
            ransac_n=3,
            num_iterations=config.clustering.iter_thresh_ransac
        )

        if len(ransac_inliers) > min_count:
            point_labels = dbscan_clustering_on_plane(
                working_points=array_working,
                plane_ids=ransac_inliers,
                source_points=array_source,
                source_labels=point_labels,
                config=config
            )


        else:
            min_count -= 1
            if min_count < config.clustering.count_thresh_ransac_rest:
                break

    return point_labels

def dbscan_clustering_on_plane(working_points, plane_ids,
                               source_points, source_labels,
                               config):
    dbscan_clustering = DBSCAN(
        eps=config.clustering.dist_thresh_dbscan,
        min_samples=config.clustering.count_thresh_dbscan
    )
    dbscan_cluster_labels = dbscan_clustering.labels_
    # check if all labels are -1
    if np.all(dbscan_cluster_labels == -1):
        return source_labels
    else:




    return point_labels