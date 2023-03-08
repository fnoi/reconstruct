import numpy as np
#from structure.CloudSegment import CloudSegment


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


def warped_vectors_intersection(seg1, seg2):
    dir1 = seg1.right - seg1.left
    dir2 = seg2.right - seg2.left
    #dir1 = seg1.pca
    #dir2 = seg2.pca
    connect = np.cross(dir1, dir2)

    if np.nonzero(connect) is False:
        raise 'Vectors are parallel'
    # kicked recently! # dist = np.abs(np.dot(seg1.center - seg2.center, connect)) / np.linalg.norm(connect)
    # source: https://math.stackexchange.com/questions/2213165/find-shortest-distance-between-lines-in-3d
    t1 = np.dot(np.cross(dir2, connect), (seg2.center - seg1.center)) / np.dot(connect, connect)
    t2 = np.dot(np.cross(dir1, connect), (seg2.center - seg1.center)) / np.dot(connect, connect)

    bridgepoint1 = seg1.center + t1 * dir1
    bridgepoint2 = seg2.center + t2 * dir2

    # check if t1 is in segment 1 and t2 in segment 2
    xrange1 = np.sort([seg1.left[0], seg1.right[0]])
    yrange1 = np.sort([seg1.left[1], seg1.right[1]])
    zrange1 = np.sort([seg1.left[2], seg1.right[2]])
    x_check = xrange1[0] <= bridgepoint1[0] <= xrange1[1]
    y_check = yrange1[0] <= bridgepoint1[1] <= yrange1[1]
    z_check = zrange1[0] <= bridgepoint1[2] <= zrange1[1]
    check1 = x_check and y_check and z_check

    xrange2 = np.sort([seg2.left[0], seg2.right[0]])
    yrange2 = np.sort([seg2.left[1], seg2.right[1]])
    zrange2 = np.sort([seg2.left[2], seg2.right[2]])
    x_check = xrange2[0] <= bridgepoint2[0] <= xrange2[1]
    y_check = yrange2[0] <= bridgepoint2[1] <= yrange2[1]
    z_check = zrange2[0] <= bridgepoint2[2] <= zrange2[1]
    check2 = x_check and y_check and z_check

    case = None
    rating = 0
    # case 1: one segment continuous: dominant
    if (check1 and not check2) or (not check1 and check2):
        if check1:  # seg1 is dominant
            rating = np.min(
                np.asarray(
                    [np.linalg.norm(bridgepoint1 - seg2.left), np.linalg.norm(bridgepoint1 - seg2.right)]
                )
            )
            case = 0

        elif check2:  # seg2 is dominant
            rating = np.min(
                np.asarray(
                    [np.linalg.norm(bridgepoint2 - seg1.left), np.linalg.norm(bridgepoint2 - seg1.right)]
                )
            )
            case = 1

    # case 2: no dominant segment
    if not check1 and not check2:
        # find mean intersection interpolated between both segments
        # report the worse rating of both segments
        rating1 = np.min(
            np.asarray(
                [np.linalg.norm(bridgepoint1 - seg2.left), np.linalg.norm(bridgepoint1 - seg2.right)]
            )
        )
        rating2 = np.min(
            np.asarray(
                [np.linalg.norm(bridgepoint2 - seg1.left), np.linalg.norm(bridgepoint2 - seg1.right)]
            )
        )
        rating = np.max(np.asarray([rating1, rating2]))
        case = 2

    # case 3: both segments continuous: intersecting
    if check1 and check2:
        rating = np.linalg.norm(bridgepoint1 - bridgepoint2)
        case = 3

    if rating == 0 or rating > 0 is False:
    # if rating > 0 is False:
        rating = 1e8

    return bridgepoint1, bridgepoint2, rating, case


# def passing_check:




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
                    if np.linalg.norm(point - bridgepoint1) < 0.1: # TODO: make this a parameter
                        print('joining on existent intermediate point')
                        bridgepoint1 = point
                        matched = True
            if not matched:
                print('adding intermediate point')
                segment1.intermediate_points.append(bridgepoint1)
        else:
            print('adding intermediate point')
            segment1.intermediate_points.append(bridgepoint1)

            if np.linalg.norm(bridgepoint1 - segment2.left) < np.linalg.norm(bridgepoint1 - segment2.right):
                segment2.left = bridgepoint1
            elif np.linalg.norm(bridgepoint1 - segment2.left) > np.linalg.norm(bridgepoint1 - segment2.right):
                segment2.right = bridgepoint1
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
                    if np.linalg.norm(point - bridgepoint2) < 0.1: # TODO: make this a parameter
                        print('joining on existent intermediate point')
                        bridgepoint2 = point
                        matched = True
            if not matched:
                print('adding intermediate point')
                segment2.intermediate_points.append(bridgepoint2)
        else:
            print('adding intermediate point')
            segment2.intermediate_points.append(bridgepoint2)

            if np.linalg.norm(bridgepoint2 - segment1.left) < np.linalg.norm(bridgepoint2 - segment1.right):
                segment1.left = bridgepoint2
            elif np.linalg.norm(bridgepoint2 - segment1.left) > np.linalg.norm(bridgepoint2 - segment1.right):
                segment1.right = bridgepoint2
            else:
                raise 'really case 1?'

    elif case == 2:
        # no dominant segment
        bridgepoint = (bridgepoint1 + bridgepoint2) / 2
        if np.linalg.norm(bridgepoint2 - segment1.left) < np.linalg.norm(bridgepoint2 - segment1.right):
            segment1.left = bridgepoint
        elif np.linalg.norm(bridgepoint2 - segment1.left) > np.linalg.norm(bridgepoint2 - segment1.right):
            segment1.right = bridgepoint
        else:
            raise 'really case 2?'

        if np.linalg.norm(bridgepoint1 - segment2.left) < np.linalg.norm(bridgepoint1 - segment2.right):
            segment2.left = bridgepoint
        elif np.linalg.norm(bridgepoint1 - segment2.left) > np.linalg.norm(bridgepoint1 - segment2.right):
            segment2.right = bridgepoint
        else:
            raise 'really case 2?'

    else:
        raise 'Case not defined'

    return segment1, segment2
