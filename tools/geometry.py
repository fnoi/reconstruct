import copy
import itertools
from typing import Tuple, Any

import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
from matplotlib import pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

import plotly.graph_objects as go

from scipy.spatial import distance
from math import degrees, acos

from sklearn import linear_model


def angle_between_planes(plane1, plane2):
    normal1 = plane1[:3]
    normal2 = plane2[:3]
    cos_angle = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
    angle = degrees(acos(cos_angle))
    return angle


def normal_and_point_to_plane(normal, point):
    plane = np.zeros(4)
    plane[:3] = normal
    plane[3] = - np.dot(normal, point)
    return plane


def intersection_point_of_line_and_plane(line_point, direction, plane):
    normal = plane[:3]
    constant = plane[3]
    denominator = np.dot(normal, direction)
    if denominator == 0:
        raise Exception("Line is parallel to plane, no intersection found")
    t = - (np.dot(normal, line_point) + constant) / denominator
    intersection_point = line_point + t * direction
    return intersection_point


def intersection_point_of_line_and_plane_rev(line_point, direction, plane):
    a, b, c, d = plane
    p_x, p_y, p_z = line_point
    v_x, v_y, v_z = direction

    denom = a * v_x + b * v_y + c * v_z

    if np.isclose(denom, 0):
        print("line is parallel to plane")
        return None

    # Solve for t
    t = -(a * p_x + b * p_y + c * p_z + d) / denom

    # Calculate the intersection point
    intersection_point = line_point + t * direction

    return intersection_point


# def intersecting_line(plane1, plane2):
#     normal1 = np.array(plane1[:3])
#     normal2 = np.array(plane2[:3])
#     d1 = plane1[3]
#     d2 = plane2[3]
#     direction = np.cross(normal1, normal2)
#     if np.linalg.norm(direction) == 0:
#         raise Exception("Planes are parallel, no intersection found")
#     point_on_line = np.linalg.solve(
#         np.column_stack((normal1, normal2, direction)), -np.array([d1, d2, 0])
#
#     )
#     return point_on_line, direction

def intersecting_line(plane1, plane2):
    normal1, d1 = np.array(plane1[:3], dtype=np.float64), plane1[3]
    normal2, d2 = np.array(plane2[:3], dtype=np.float64), plane2[3]

    # Normalize the normals to ensure calculations are based on unit vectors
    normal1 = normal1 / np.linalg.norm(normal1)
    normal2 = normal2 / np.linalg.norm(normal2)

    # Calculate the direction of the line of intersection
    direction = np.cross(normal1, normal2)

    # Check if the planes are parallel or coincident (cross product is zero)
    if np.linalg.norm(direction) < 1e-10:  # Use a small threshold to handle floating-point precision issues
        print("The planes are parallel or coincident. No unique line of intersection.")
        return None, None

    # Calculate an origin point on the line of intersection
    try:
        origin = np.cross((normal1 * d2 - normal2 * d1), direction) / np.linalg.norm(direction) ** 2
    except FloatingPointError:
        print("Numerical stability issue encountered while calculating the origin.")
        return None, None

    return direction, origin


def line_of_intersection(plane1, plane2):
    normal1 = plane1[:3]
    normal2 = plane2[:3]
    direction = np.cross(normal1, normal2)
    point_on_line = None

    # Find a point on the line by checking if any of the coordinates of the cross product are non-zero
    for i in range(3):
        if direction[i] != 0:
            constant = - plane1[3] / plane1[i]
            point_on_line = np.zeros(3)
            point_on_line[i] = constant
            break

    if point_on_line is None:
        raise Exception("Planes are parallel, no intersection found")

    return point_on_line, direction


def intersecting_line_between_planes(plane1, plane2):
    angle = angle_between_planes(plane1, plane2)
    if 45 < angle < 235:
        return line_of_intersection(plane1, plane2)
    else:
        raise Exception(f"Angle between planes is {angle}, which is not between 45 and 235 degrees")


def rotation_matrix_from_vectors(vec1, vec2):
    """Return matrix to rotate one vector to another.

    Parameters
    ----------
    vec1 : array-like
        Vector to rotate.
    vec2 : array-like
        Vector to rotate to.

    Returns
    -------
    R : array-like
        Rotation matrix.

    """

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    v = np.cross(vec1, vec2)
    s = np.linalg.norm(v)
    c = np.dot(vec1, vec2)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / s ** 2)
    return R


def skew_lines(seg1, seg2):
    ptc_1 = np.mean([seg1.left_3D, seg1.right_3D], axis=0)
    ptc_2 = np.mean([seg2.left_3D, seg2.right_3D], axis=0)
    dir_1 = seg1.right_3D - seg1.left_3D
    dir_2 = seg2.right_3D - seg2.left_3D

    connect = ptc_2 - ptc_1
    dir_1_dot_dir_2 = np.dot(dir_1, dir_2)

    if np.isclose(dir_1_dot_dir_2, 1) or np.isclose(dir_1_dot_dir_2, -1):
        print('Lines are parallel')
        return None, None, None, None, 0

    else:

        A = np.array([
            [np.dot(dir_1, dir_1), -dir_1_dot_dir_2],
            [dir_1_dot_dir_2, -np.dot(dir_2, dir_2)]
        ])

        B = np.array([
            np.dot(connect, dir_1),
            np.dot(connect, dir_2)
        ])

        t1, t2 = np.linalg.solve(A, B)

        bridgepoint_1 = ptc_1 + t1 * dir_1
        bridgepoint_2 = ptc_2 + t2 * dir_2

        distance = np.linalg.norm(bridgepoint_1 - bridgepoint_2)

        angle = np.arccos(
            np.clip(
                np.dot(dir_1, dir_2) / (np.linalg.norm(dir_1) * np.linalg.norm(dir_2)), -1.0, 1.0
            )
        )

        angle_deg = np.rad2deg(angle)

        nearest_point = (bridgepoint_1 + bridgepoint_2) / 2

        within_seg1 = (np.dot(bridgepoint_1 - seg1.left_3D, seg1.right_3D - seg1.left_3D) >= 0 >= np.dot(bridgepoint_1 - seg1.right_3D, seg1.right_3D - seg1.left_3D))
        within_seg2 = (np.dot(bridgepoint_2 - seg2.left_3D, seg2.right_3D - seg2.left_3D) >= 0 >= np.dot(bridgepoint_2 - seg2.right_3D, seg2.right_3D - seg2.left_3D))

        if within_seg1 and not within_seg2:
            case = 0
        elif within_seg2 and not within_seg1:
            case = 1
        elif not within_seg1 and not within_seg2:
            case = 2
        else:
            case = 3

        debug_plot = False
        # if distance < 0.3:
        #     debug_plot = True
        # if not debug_plot:
        #     print(f'Rating: {distance}, Case: {case}, Angle: {angle}')
        if debug_plot:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=[seg1.left_3D[0], seg1.right_3D[0]],
                y=[seg1.left_3D[1], seg1.right_3D[1]],
                z=[seg1.left_3D[2], seg1.right_3D[2]],
                mode='lines',
                line=dict(
                    color='blue',
                    width=3
                )
            ))
            fig.add_trace(go.Scatter3d(
                x=[seg2.left_3D[0], seg2.right_3D[0]],
                y=[seg2.left_3D[1], seg2.right_3D[1]],
                z=[seg2.left_3D[2], seg2.right_3D[2]],
                mode='lines',
                line=dict(
                    color='red',
                    width=3
                )
            ))
            fig.add_trace(go.Scatter3d(
                x=[bridgepoint_1[0]],
                y=[bridgepoint_1[1]],
                z=[bridgepoint_1[2]],
                mode='markers',
                marker=dict(
                    size=5,
                    color='green',
                    opacity=1
                )
            ))
            fig.add_trace(go.Scatter3d(
                x=[bridgepoint_2[0]],
                y=[bridgepoint_2[1]],
                z=[bridgepoint_2[2]],
                mode='markers',
                marker=dict(
                    size=5,
                    color='violet',
                    opacity=1
                )
            ))
            # title with rating, type of case and angle
            fig.update_layout(title=f'Rating: {distance}, Case: {case}, Angle: {angle_deg}')
            fig.show()

        return bridgepoint_1, bridgepoint_2, distance, case, angle_deg


def warped_vectors_intersection(seg1, seg2):
    # dir1 = seg1.right_3D - seg1.left_3D
    # dir2 = seg2.right_3D - seg2.left_3D
    # dir1 = seg1.pca
    # dir2 = seg2.pca
    dir1 = seg1.right_3D - seg1.left_3D
    dir2 = seg2.right_3D - seg2.left_3D

    # calculate angle of "intersection"
    angle = np.arccos(np.clip(np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2)), -1, 1))
    angle = np.degrees(angle)

    connect = np.cross(dir1, dir2)

    if np.nonzero(connect) is False:
        raise 'Vectors are parallel'
    # kicked recently! # dist = np.abs(np.dot(seg1.center - seg2.center, connect)) / np.linalg.norm(connect)
    # source: https://math.stackexchange.com/questions/2213165/find-shortest-distance-between-lines-in-3d

    if seg1.points_center is None:
        seg1.points_center = (seg1.left_3D + seg1.right_3D) / 2
    if seg2.points_center is None:
        seg2.points_center = (seg2.left_3D + seg2.right_3D) / 2

    connect_2 = seg2.points_center - seg1.points_center

    A = np.array([
        [np.dot(dir1, dir1), -np.dot(dir1, dir2)],
        [np.dot(dir1, dir2), -np.dot(dir2, dir2)]
    ])

    B = np.array([
        np.dot(connect_2, dir1),
        np.dot(connect_2, dir2)
    ])

    t1_2, t2_2 = np.linalg.solve(A, B)

    bridgepoint1_2 = seg1.points_center + t1_2 * dir1
    bridgepoint2_2 = seg2.points_center + t2_2 * dir2



    t1 = np.dot(np.cross(dir2, connect), (seg2.points_center - seg1.points_center)) / np.dot(connect, connect)
    t2 = np.dot(np.cross(dir1, connect), (seg2.points_center - seg1.points_center)) / np.dot(connect, connect)

    bridgepoint1 = seg1.points_center + t1 * dir1
    bridgepoint2 = seg2.points_center + t2 * dir2

    bridgepoint1 = bridgepoint1_2
    bridgepoint2 = bridgepoint2_2

    # check if t1 is in segment 1 and t2 in segment 2
    xrange1 = np.sort([seg1.left_3D[0], seg1.right_3D[0]])
    yrange1 = np.sort([seg1.left_3D[1], seg1.right_3D[1]])
    zrange1 = np.sort([seg1.left_3D[2], seg1.right_3D[2]])
    x_check = xrange1[0] <= bridgepoint1[0] <= xrange1[1]
    y_check = yrange1[0] <= bridgepoint1[1] <= yrange1[1]
    z_check = zrange1[0] <= bridgepoint1[2] <= zrange1[1]
    check1 = x_check and y_check and z_check

    xrange2 = np.sort([seg2.left_3D[0], seg2.right_3D[0]])
    yrange2 = np.sort([seg2.left_3D[1], seg2.right_3D[1]])
    zrange2 = np.sort([seg2.left_3D[2], seg2.right_3D[2]])
    x_check = xrange2[0] <= bridgepoint2[0] <= xrange2[1]
    y_check = yrange2[0] <= bridgepoint2[1] <= yrange2[1]
    z_check = zrange2[0] <= bridgepoint2[2] <= zrange2[1]
    check2 = x_check and y_check and z_check

    case = None
    rating = 0
    # case 1: one segment continuous: dominant
    if (check1 and not check2) or (not check1 and check2):
        if check1:  # seg1 is dominant: case = 0
            rating = np.min(
                np.asarray(
                    [np.linalg.norm(bridgepoint1 - seg2.left_3D), np.linalg.norm(bridgepoint1 - seg2.right_3D)]
                )
            )
            case = 0

        elif check2:  # seg2 is dominant: case = 1
            rating = np.min(
                np.asarray(
                    [np.linalg.norm(bridgepoint2 - seg1.left_3D), np.linalg.norm(bridgepoint2 - seg1.right_3D)]
                )
            )
            case = 1

    # case 2: no dominant segment: case = 2
    if not check1 and not check2:
        # find mean intersection interpolated between both segments
        # report the worse rating of both segments
        rating1 = np.min(
            np.asarray(
                [np.linalg.norm(bridgepoint1 - seg2.left_3D), np.linalg.norm(bridgepoint1 - seg2.right_3D)]
            )
        )
        rating2 = np.min(
            np.asarray(
                [np.linalg.norm(bridgepoint2 - seg1.left_3D), np.linalg.norm(bridgepoint2 - seg1.right_3D)]
            )
        )
        rating = np.max(np.asarray([rating1, rating2]))
        case = 2

    # case 3: both segments continuous: intersecting
    if check1 and check2:
        rating = np.linalg.norm(bridgepoint1 - bridgepoint2)
        case = 3

    if rating == 0:
        rating = 1e8
    # if rating == 0 or rating > 0 == False:
        # if rating > 0 is False:
        # rating = 1e8

    return bridgepoint1_2, bridgepoint2_2, rating, case, angle


# def passing_check:


def rotate_points_3D(points, angle, axis):
    axis_normalized = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2)
    b, c, d = -axis_normalized * np.sin(angle / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return np.dot(points, rotation_matrix)


def rotate_points_2D(points, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return np.dot(points, rotation_matrix)


def manipulate_skeleton(segment1, segment2,
                        bridgepoint1: np.ndarray, bridgepoint2: np.ndarray,
                        case: int):
    if case == 0:
        # segment 1 is dominant
        if len(segment1.intermediate_points) > 0:
            matched = False
            for point in segment1.intermediate_points:
                if matched:
                    break
                else:
                    if np.linalg.norm(point - bridgepoint1) < 0.1:  # TODO: make this a parameter
                        print('joining on existent intermediate point')
                        bridgepoint1 = point
                        matched = True
            if not matched:
                print('adding intermediate point')
                segment1.intermediate_points.append(bridgepoint1)
        else:
            print('adding intermediate point')
            segment1.intermediate_points.append(bridgepoint1)

            if np.linalg.norm(bridgepoint1 - segment2.line_raw_left) < np.linalg.norm(bridgepoint1 - segment2.line_raw_right):
                segment2.line_raw_left = bridgepoint1
            elif np.linalg.norm(bridgepoint1 - segment2.line_raw_left) > np.linalg.norm(bridgepoint1 - segment2.line_raw_right):
                segment2.line_raw_right = bridgepoint1
            else:
                raise 'really case 0?'

    elif case == 1:
        # segment 2 is dominant
        if len(segment2.intermediate_points) > 0:
            matched = False
            for point in segment2.intermediate_points:
                if matched:
                    break
                else:
                    if np.linalg.norm(point - bridgepoint2) < 0.1:  # TODO: make this a parameter
                        print('joining on existent intermediate point')
                        bridgepoint2 = point
                        matched = True
            if not matched:
                print('adding intermediate point')
                segment2.intermediate_points.append(bridgepoint2)
        else:
            print('adding intermediate point')
            segment2.intermediate_points.append(bridgepoint2)

            if np.linalg.norm(bridgepoint2 - segment1.line_raw_left) < np.linalg.norm(bridgepoint2 - segment1.line_raw_right):
                segment1.line_raw_left = bridgepoint2
            elif np.linalg.norm(bridgepoint2 - segment1.line_raw_left) > np.linalg.norm(bridgepoint2 - segment1.line_raw_right):
                segment1.line_raw_right = bridgepoint2
            else:
                raise 'really case 1?'

    elif case == 2:
        # no dominant segment
        bridgepoint = (bridgepoint1 + bridgepoint2) / 2
        if np.linalg.norm(bridgepoint2 - segment1.line_raw_left) < np.linalg.norm(bridgepoint2 - segment1.line_raw_right):
            segment1.line_raw_left = bridgepoint
        elif np.linalg.norm(bridgepoint2 - segment1.line_raw_left) > np.linalg.norm(bridgepoint2 - segment1.line_raw_right):
            segment1.line_raw_right = bridgepoint
        else:
            raise 'really case 2?'

        if np.linalg.norm(bridgepoint1 - segment2.line_raw_left) < np.linalg.norm(bridgepoint1 - segment2.line_raw_right):
            segment2.line_raw_left = bridgepoint
        elif np.linalg.norm(bridgepoint1 - segment2.line_raw_left) > np.linalg.norm(bridgepoint1 - segment2.line_raw_right):
            segment2.line_raw_right = bridgepoint
        else:
            raise 'really case 2?'

    else:
        raise 'Case not defined'

    return segment1, segment2


def project_points_onto_plane(points, normal):
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Calculate the projections
    projections = points - np.dot(points, normal)[:, np.newaxis] * normal
    return projections


def points_to_actual_plane(points, normal, point_on_plane):
    # normalize normal vector
    normal = normal / np.linalg.norm(normal)

    # calculate the projections
    projections = points - np.dot(points - point_on_plane, normal)[:, np.newaxis] * normal

    # aggregate rotation matrix
    R = rotation_matrix_from_vectors(normal, np.array([0, 0, 1]))

    return projections


def project_points_to_plane(points, plane_normal, point_on_plane):
    plane_normal_normalized = plane_normal / np.linalg.norm(plane_normal)
    vec_to_points = points - point_on_plane
    scalar_proj = np.dot(vec_to_points, plane_normal_normalized)
    vec_proj = np.outer(scalar_proj, plane_normal_normalized)
    orthogonal_vec = vec_to_points - vec_proj
    projected_points = points - orthogonal_vec
    return projected_points


def project_points_to_line(points, point_on_line, direction):
    """
    Project points to a line defined by a point on the line and the direction of the line
    Parameters
    ----------
    points: ndarray of shape (n, 3)
        Points to project to the line
    point_on_line: ndarray of shape (3,)
        A point on the line
    direction: ndarray of shape (3,)
        Direction of the line
    -------
    """

    direction_normalized = direction / np.linalg.norm(direction)
    vec_to_points = points - point_on_line
    scalar_proj = np.dot(vec_to_points, direction_normalized)[:, np.newaxis] * direction_normalized
    # scalar_proj = np.dot(vec_to_points, direction_normalized)
    # vec_proj = np.outer(scalar_proj, direction_normalized)
    # projected_points = point_on_line + vec_proj
    # calculate distance of points to line
    dists = np.linalg.norm(vec_to_points - scalar_proj, axis=1)
    closest_ind = np.argmin(dists)

    projected_points = scalar_proj + point_on_line
    return projected_points, closest_ind


def rotation_matrix_from_vectors(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    axis = np.cross(vec1, vec2)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-7:
        return np.identity(3)  # No rotation needed
    axis = axis / axis_norm

    angle = np.arccos(np.dot(vec1, vec2))

    kmat = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])

    return np.identity(3) + np.sin(angle) * kmat + (1 - np.cos(angle)) * np.dot(kmat, kmat)



def orientation_estimation(cluster_ptx_array, config=None, step=None):
    """takes in xyz array of points, performs ransac until 2 non-planar planes are found
    then returns vector describing the line of intersection between the two planes"""

    case = config.skeleton.ransac_method
    if case == "open3d":
        # convert to open3d point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(cluster_ptx_array[:, :3])
        point_cloud.normals = o3d.utility.Vector3dVector(cluster_ptx_array[:, 3:6])
        # perform ransac
        if step == "skeleton":
            dist_threshold = config.skeleton.ransac_distance_threshold
            ransac_n = config.skeleton.ransac_ransac_n
            num_iterations = config.skeleton.ransac_num_iterations
            prob = 1.0
        else:
            dist_threshold = 0.01
            ransac_n = 3
            num_iterations = 10000000
            prob = 0.9999

        f_0, inliers_0 = point_cloud.segment_plane(
            distance_threshold=dist_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
            probability=prob
        )
        # remove inliers from point cloud
        point_cloud = point_cloud.select_by_index(inliers_0, invert=True)
        while True:
            # perform ransac again
            f_1, inliers_1 = point_cloud.segment_plane(
                distance_threshold=dist_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            angle = np.rad2deg(
                np.arccos(
                    np.dot(
                        f_0[:3],
                        f_1[:3]
                    )
                )
            )
            if 45 < angle % 180 < 135:
                inlier_coords = np.asarray(point_cloud.points)[inliers_1]
                # find the corresponding id in cluster_ptx_array for every row
                inliers_1_fix = np.where(np.isin(cluster_ptx_array[:, :3], inlier_coords).all(axis=1))[0]
                inliers_1 = inliers_1_fix.tolist()

                print('ok')
                break
            else:
                # print('.../...\...') #planes found are not "perpendicular" enough, retrying...')
                point_cloud = point_cloud.select_by_index(inliers_1, invert=True)
                # if not enough points left, break
                if len(point_cloud.points) < 0.1 * len(cluster_ptx_array):
                    print('fail')
                    return None, None, None, None, None


    elif case == "pyransac":
        plane = pyrsc.Plane()
        planes = []
        points = copy.deepcopy(cluster_ptx_array[:, :3])
        while True:
            ransac_result = plane.fit(pts=points,
                                      thresh=config.skeleton.ransac_dist_thresh,
                                      minPoints=config.skeleton.ransac_min_count_rel * len(points),
                                      maxIteration=config.skeleton.ransac_iterations)
            planes.append(ransac_result[0])
            points = np.delete(points, ransac_result[1], axis=0)
            if len(planes) > 1:
                plane_combinations = itertools.combinations(range(len(planes)), 2)
                for combination in plane_combinations:
                    f_0 = planes[combination[0]]
                    f_1 = planes[combination[1]]

                    angle = np.rad2deg(
                        np.arccos(
                            np.dot(
                                f_0[:3],
                                f_1[:3]
                            )
                        )
                    )
                    if 45 < angle % 180 < 135:
                        break
                raise Exception('planes found are not "perpendicular" enough, cannot retry ...')

    else:
        raise Exception('ransac_method not recognized')

    normal1, d1 = np.array(f_0[:3], dtype=np.float64), f_0[3]
    normal2, d2 = np.array(f_1[:3], dtype=np.float64), f_1[3]
    orientation = np.cross(normal1, normal2)

    # A = np.array([normal1, normal2, orientation])
    # B = np.array([-d1, -d2, 0])
    # point_on_line = np.linalg.lstsq(A.T, B, rcond=None)[0]
    point_on_line = np.cross((normal1 * d2 - normal2 * d1), orientation) / np.linalg.norm(orientation) ** 2

    if step == "skeleton":
        return (f_0, f_1), orientation, point_on_line, inliers_0, inliers_1
    else:
        return orientation


def orientation_2D(cloud):
    coords = copy.deepcopy(cloud.points_flat_raw)
    active_mask = np.ones(len(coords), dtype=bool)
    lines = []
    # perform ransac in 2D to find all lines using scipy
    while True:
        # ransac for line with pyrsc
        line = pyrsc.Line()
        ransac_result = line.fit(pts=coords[active_mask],
                                 thresh=0.01,
                                 maxIteration=100000)
        lines.append(ransac_result[0])
        # remove inliers from point cloud
        inliers = ransac_result[1]
        active_mask[inliers] = False

        fig, ax = plt.subplots()
        ax.scatter(coords[:, 0], coords[:, 1], s=1)
        for line in lines:
            x = np.linspace(np.min(coords[:, 0]), np.max(coords[:, 0]), 3)
            y = (-line[0] * x - line[2]) / line[1]
            ax.plot(x, y, color='red')
        plt.show()

        a = 0

        if len(coords[active_mask]) < 0.1 * len(coords):
            break


def rotate_xy2xyz(point_2D, rot_matrix, xy_angle):
    rot_mat = np.asarray(rot_matrix)
    z_angle_add = xy_angle
    rot_mat_z = np.asarray([[np.cos(z_angle_add), -np.sin(z_angle_add), 0],
                            [np.sin(z_angle_add), np.cos(z_angle_add), 0],
                            [0, 0, 1]])
    rot_mat = np.dot(rot_mat.T, rot_mat_z)

    point_3D = np.array(np.dot(
        np.array([point_2D[0], point_2D[1], 0]),
        rot_mat.T
    ))

    return point_3D


def angle_between_line_segments(origin, endpoint_0, endpoint_1):
    """compute angle between two lines defined by shared origin and endpoints"""
    v0 = np.array(endpoint_0) - np.array(origin)
    v1 = np.array(endpoint_1) - np.array(origin)

    dot_product = np.dot(v0, v1)

    v0_mag = np.linalg.norm(v0)
    v1_mag = np.linalg.norm(v1)

    cos_angle = dot_product / (v0_mag * v1_mag)

    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    cross_product = np.cross(v0, v1)
    if cross_product < 0:
        angle_rad = 2 * np.pi - angle_rad

    return angle_rad


def transform_up(rotation_pose, rotation_long, translation):
    mat = np.eye(4)
    mat[:3, :3] = rotation_pose
    mat[:3, 3] = translation

    return mat

def transform_down(rotation_pose, rotation_long, translation):
    mat = np.eye(4)
    mat[:3, :3] = rotation_pose.T
    mat[:3, 3] = -np.dot(rotation_pose.T, translation)

    return mat


def transform_lines(line_0_left, line_0_right, line_1_left, line_1_right):
    line_0_left, line_0_right, line_1_left, line_1_right = map(
        np.array,
        [line_0_left, line_0_right, line_1_left, line_1_right]
    )

    # Calculate vectors
    vec_0 = line_0_right - line_0_left
    vec_1 = line_1_right - line_1_left

    # Normalize vectors
    vec_0_norm = vec_0 / np.linalg.norm(vec_0)
    vec_1_norm = vec_1 / np.linalg.norm(vec_1)

    # Calculate rotation
    rotation_axis = np.cross(vec_0_norm, vec_1_norm)
    if np.allclose(rotation_axis, 0):
        if np.allclose(vec_0_norm, vec_1_norm):
            rot_mat = np.eye(3)
        else:
            rot_mat = -np.eye(3)
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        cos_angle = np.dot(vec_0_norm, vec_1_norm)
        sin_angle = np.sqrt(1 - cos_angle ** 2)  # More numerically stable than np.linalg.norm(np.cross())

        # Rodrigues' rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        rot_mat = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)

    # Calculate scaling factor
    scale = np.linalg.norm(vec_1) / np.linalg.norm(vec_0)

    # Apply scaling to rotation matrix
    rot_mat = scale * rot_mat

    # Calculate translation
    translation = line_1_left - np.dot(rot_mat, line_0_left)

    # Construct 4x4 transformation matrix
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rot_mat
    trans_mat[:3, 3] = translation

    # Debug information
    print("Debug Information:")
    print(f"vec_0: {vec_0}, length: {np.linalg.norm(vec_0)}")
    print(f"vec_1: {vec_1}, length: {np.linalg.norm(vec_1)}")
    print(f"Scaling factor: {scale}")
    print(f"Rotation matrix:\n{rot_mat}")
    print(f"Translation: {translation}")
    print(f"Final transformation matrix:\n{trans_mat}")

    # Verify transformation
    transformed_line_0_left = np.dot(trans_mat, np.append(line_0_left, 1))[:3]
    transformed_line_0_right = np.dot(trans_mat, np.append(line_0_right, 1))[:3]
    print("\nVerification:")
    print(f"Transformed line_0_left: {transformed_line_0_left}")
    print(f"Original line_1_left: {line_1_left}")
    print(f"Transformed line_0_right: {transformed_line_0_right}")
    print(f"Original line_1_right: {line_1_right}")

    return trans_mat


def add_z_rotation_angle(transform_matrix, angle_degrees, target_point):
    """
    Incorporate a rotation around the z-axis at the target point into the existing transformation matrix.

    :param transform_matrix: The original 4x4 transformation matrix
    :param angle_degrees: The rotation angle in degrees
    :param target_point: The 3D point around which to rotate
    :return: A new 4x4 transformation matrix incorporating the z-axis rotation
    """
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Create the z-axis rotation matrix
    cos_theta, sin_theta = np.cos(angle_radians), np.sin(angle_radians)
    z_rotation = np.array([
        [cos_theta, -sin_theta, 0, 0],
        [sin_theta, cos_theta, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Create translation matrices
    translate_to_origin = np.eye(4)
    translate_to_origin[:3, 3] = -target_point

    translate_back = np.eye(4)
    translate_back[:3, 3] = target_point

    # Combine all transformations
    final_transform = np.dot(translate_back, np.dot(z_rotation, np.dot(translate_to_origin, transform_matrix)))

    return final_transform


def add_z_rotation_matrix(transform_matrix, rotation_matrix, target_point):
    """
    Incorporate a rotation described by a 3x3 rotation matrix around the z-axis at the target point
    into the existing transformation matrix.

    :param transform_matrix: The original 4x4 transformation matrix
    :param rotation_matrix: A 3x3 rotation matrix describing the z-axis rotation
    :param target_point: The 3D point around which to rotate
    :return: A new 4x4 transformation matrix incorporating the z-axis rotation
    """
    # Check if the rotation matrix is valid
    if rotation_matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3")

    # Create a 4x4 transformation matrix from the 3x3 rotation matrix
    z_rotation = np.eye(4)
    z_rotation[:3, :3] = rotation_matrix

    # Create translation matrices
    translate_to_origin = np.eye(4)
    translate_to_origin[:3, 3] = -target_point

    translate_back = np.eye(4)
    translate_back[:3, 3] = target_point

    # Combine all transformations
    final_transform = np.dot(translate_back, np.dot(z_rotation, np.dot(translate_to_origin, transform_matrix)))

    return final_transform


def simplified_transform_lines(source_angle, target_angle):
    """
    Calculate the transformation matrix to map one normalized right angle to another.

    :param source_angle: Tuple of (left, common, right) points for the source angle
    :param target_angle: Tuple of (left, common, right) points for the target angle
    :return: 4x4 transformation matrix
    """
    src_left, src_common, src_right = map(np.array, source_angle)
    tgt_left, tgt_common, tgt_right = map(np.array, target_angle)

    # Calculate normalized vectors for source and target angles
    src_x = src_left - src_common
    src_y = src_right - src_common
    src_z = np.cross(src_x, src_y)

    tgt_x = tgt_left - tgt_common
    tgt_y = tgt_right - tgt_common
    tgt_z = np.cross(tgt_x, tgt_y)

    # Create rotation matrices
    rot_src = np.column_stack((src_x, src_y, src_z))
    rot_tgt = np.column_stack((tgt_x, tgt_y, tgt_z))

    # Calculate the rotation from source to target
    rotation = np.dot(rot_tgt, rot_src.T)

    # Calculate the translation
    translation = tgt_common - np.dot(rotation, src_common)

    # Create the 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    debug = False
    if debug:
        # Verification
        print("Verification:")
        for src, tgt, label in zip([src_left, src_common, src_right],
                                   [tgt_left, tgt_common, tgt_right],
                                   ['Left', 'Common', 'Right']):
            transformed = np.dot(transform, np.append(src, 1))[:3]
            print(f"{label} point:")
            print(f"  Original:    {src}")
            print(f"  Transformed: {transformed}")
            print(f"  Target:      {tgt}")
            print(f"  Error:       {np.linalg.norm(transformed - tgt)}")
            print()

    return transform


def calculate_shifted_source_pt(angle_source, shift_x, shift_y, third_pt=None):
    # calculate global direction of y as cross product of source angle vectors
    vec_y = angle_source[0] - angle_source[1]  # local y direction
    vec_z = angle_source[2] - angle_source[1] # local z direction
    vec_x = np.cross(vec_y, vec_z)  # local x direction
    vec_x = vec_x / np.linalg.norm(vec_x)  # normalize

    if third_pt is None:
        common = angle_source[1]
    else:
        common = third_pt

    shifted_pt = common + shift_x * vec_x + shift_y * vec_y

    return shifted_pt




def calculate_shifted_point(angle_tuple, shift_x, shift_y):
    """
    Calculate a new point based on x,y shifts in the plane perpendicular to the angle's normal.
    The angle vectors define the y-z plane, and shifts are applied in the x-y plane.

    :param angle_tuple: Tuple of (left, common, right) points defining the angle in y-z
    :param shift_x: Shift in global x direction
    :param shift_y: Shift in direction of 'left' vector projection onto x-y plane
    :return: The new point coordinates in 3D space
    """
    left, common, right = map(np.array, angle_tuple)

    # Get normalized direction vectors
    vec_left = left - common  # first angle vector (in y-z)
    vec_right = right - common  # second angle vector (in y-z)

    # Calculate normal of the y-z plane defined by the angle
    normal = np.cross(vec_left, vec_right)
    normal = normal / np.linalg.norm(normal)  # ensure normalized

    # Project left vector onto x-y plane
    # First, get the component of vec_left perpendicular to normal
    proj_left = vec_left - np.dot(vec_left, normal) * normal
    # Then project this onto x-y plane
    proj_left[2] = 0  # zero out z component
    if np.linalg.norm(proj_left) > 0:
        proj_left = proj_left / np.linalg.norm(proj_left)  # normalize

    # x direction is simply the global x direction
    x_dir = np.array([1, 0, 0])

    # Calculate the shifted point
    shifted_point = common + shift_x * x_dir + shift_y * proj_left

    return shifted_point


def transform_shifted_point(source_angle, target_angle, local_point):
    """
    Transform a point from source angle's local coordinate system to target system.

    :param source_angle: Tuple of (left, common, right) points for source
    :param target_angle: Tuple of (left, common, right) points for target
    :param local_point: Point in 3D space relative to source angle
    :return: Transformed point in target coordinate system
    """
    # Calculate transformation matrix
    transform = simplified_transform_lines(source_angle, target_angle)

    # Transform the point
    point_homogeneous = np.append(local_point, 1)
    transformed_point = np.dot(transform, point_homogeneous)[:3]

    return transformed_point


def kmeans_points_normals_2D(points, point_normals, n_representatives):
    points = np.array(points)
    point_normals = np.array(point_normals)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_representatives, random_state=42)
    labels = kmeans.fit_predict(points)

    # Get cluster centers as representatives
    representatives = kmeans.cluster_centers_

    # Calculate weights based on cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    weights = counts # / len(points)

    # Calculate representative normals for each cluster
    rep_normals = np.zeros((n_representatives, 2))
    for i in range(n_representatives):
        cluster_mask = labels == i
        cluster_normals = point_normals[cluster_mask]
        # Weighted average of normals
        mean_normal = np.mean(cluster_normals, axis=0)
        # Normalize to unit vector
        rep_normals[i] = mean_normal / np.linalg.norm(mean_normal)

    return representatives, rep_normals, weights, labels