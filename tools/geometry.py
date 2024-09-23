import copy
import itertools
from typing import Tuple, Any

import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
from matplotlib import pyplot as plt
from numpy import ndarray, dtype, object_
# from structure.CloudSegment import CloudSegment

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
    ptc_1 = np.mean([seg1.line_cog_left, seg1.line_cog_right], axis=0)
    ptc_2 = np.mean([seg2.line_cog_left, seg2.line_cog_right], axis=0)
    dir_1 = seg1.line_cog_right - seg1.line_cog_left
    dir_2 = seg2.line_cog_right - seg2.line_cog_left

    connect = ptc_2 - ptc_1
    dir_1_dot_dir_2 = np.dot(dir_1, dir_2)

    if np.isclose(dir_1_dot_dir_2, 1) or np.isclose(dir_1_dot_dir_2, -1):
        print('Lines are parallel')

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

    within_seg1 = (np.dot(bridgepoint_1 - seg1.line_cog_left, seg1.line_cog_right - seg1.line_cog_left) >= 0 >= np.dot(bridgepoint_1 - seg1.line_cog_right, seg1.line_cog_right - seg1.line_cog_left))
    within_seg2 = (np.dot(bridgepoint_2 - seg2.line_cog_left, seg2.line_cog_right - seg2.line_cog_left) >= 0 >= np.dot(bridgepoint_2 - seg2.line_cog_right, seg2.line_cog_right - seg2.line_cog_left))

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
            x=[seg1.line_cog_left[0], seg1.line_cog_right[0]],
            y=[seg1.line_cog_left[1], seg1.line_cog_right[1]],
            z=[seg1.line_cog_left[2], seg1.line_cog_right[2]],
            mode='lines',
            line=dict(
                color='blue',
                width=3
            )
        ))
        fig.add_trace(go.Scatter3d(
            x=[seg2.line_cog_left[0], seg2.line_cog_right[0]],
            y=[seg2.line_cog_left[1], seg2.line_cog_right[1]],
            z=[seg2.line_cog_left[2], seg2.line_cog_right[2]],
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
    # dir1 = seg1.line_cog_right - seg1.line_cog_left
    # dir2 = seg2.line_cog_right - seg2.line_cog_left
    # dir1 = seg1.pca
    # dir2 = seg2.pca
    dir1 = seg1.line_cog_right - seg1.line_cog_left
    dir2 = seg2.line_cog_right - seg2.line_cog_left

    # calculate angle of "intersection"
    angle = np.arccos(np.clip(np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2)), -1, 1))
    angle = np.degrees(angle)

    connect = np.cross(dir1, dir2)

    if np.nonzero(connect) is False:
        raise 'Vectors are parallel'
    # kicked recently! # dist = np.abs(np.dot(seg1.center - seg2.center, connect)) / np.linalg.norm(connect)
    # source: https://math.stackexchange.com/questions/2213165/find-shortest-distance-between-lines-in-3d

    if seg1.points_center is None:
        seg1.points_center = (seg1.line_cog_left + seg1.line_cog_right) / 2
    if seg2.points_center is None:
        seg2.points_center = (seg2.line_cog_left + seg2.line_cog_right) / 2

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
    xrange1 = np.sort([seg1.line_cog_left[0], seg1.line_cog_right[0]])
    yrange1 = np.sort([seg1.line_cog_left[1], seg1.line_cog_right[1]])
    zrange1 = np.sort([seg1.line_cog_left[2], seg1.line_cog_right[2]])
    x_check = xrange1[0] <= bridgepoint1[0] <= xrange1[1]
    y_check = yrange1[0] <= bridgepoint1[1] <= yrange1[1]
    z_check = zrange1[0] <= bridgepoint1[2] <= zrange1[1]
    check1 = x_check and y_check and z_check

    xrange2 = np.sort([seg2.line_cog_left[0], seg2.line_cog_right[0]])
    yrange2 = np.sort([seg2.line_cog_left[1], seg2.line_cog_right[1]])
    zrange2 = np.sort([seg2.line_cog_left[2], seg2.line_cog_right[2]])
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
                    [np.linalg.norm(bridgepoint1 - seg2.line_cog_left), np.linalg.norm(bridgepoint1 - seg2.line_cog_right)]
                )
            )
            case = 0

        elif check2:  # seg2 is dominant: case = 1
            rating = np.min(
                np.asarray(
                    [np.linalg.norm(bridgepoint2 - seg1.line_cog_left), np.linalg.norm(bridgepoint2 - seg1.line_cog_right)]
                )
            )
            case = 1

    # case 2: no dominant segment: case = 2
    if not check1 and not check2:
        # find mean intersection interpolated between both segments
        # report the worse rating of both segments
        rating1 = np.min(
            np.asarray(
                [np.linalg.norm(bridgepoint1 - seg2.line_cog_left), np.linalg.norm(bridgepoint1 - seg2.line_cog_right)]
            )
        )
        rating2 = np.min(
            np.asarray(
                [np.linalg.norm(bridgepoint2 - seg1.line_cog_left), np.linalg.norm(bridgepoint2 - seg1.line_cog_right)]
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


def rotate_points_to_xy_plane(points, normal):
    # find rotation matrix
    rot_matrix = rotation_matrix_from_vectors(normal, np.array([0, 0, 1]))
    # rotate points
    rotated_points = np.dot(points, rot_matrix.T)

    return rotated_points, rot_matrix


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

    A = np.array([normal1, normal2, orientation])
    B = np.array([-d1, -d2, 0])
    point_on_line = np.linalg.lstsq(A.T, B, rcond=None)[0]
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


