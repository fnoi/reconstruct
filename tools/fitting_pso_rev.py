import time

import numpy as np
from matplotlib import pyplot as plt

from pyswarm import pso


def point_to_line_distance(point, v1, v2):
    """Calculate the minimum distance from a point to a line segment defined by vertices v1 and v2."""
    line_vec = v2 - v1
    point_vec = point - v1
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    t = np.clip(t, 0, 1)
    nearest = v1 + t * line_vec
    dist = np.linalg.norm(nearest - point)
    return dist


def min_distance_to_polygon(points, vertices):
    """Calculate the minimum distance from each point in 'points' to a polygon defined by 'vertices'."""
    num_vertices = len(vertices)
    num_points = len(points)
    min_distances = np.inf * np.ones(num_points)

    for i in range(num_vertices):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % num_vertices]  # Wrap around to connect the last vertex to the first
        for j in range(num_points):
            point = points[j, :2]  # Assuming points are in nx3, ignore the third dimension if present
            dist = point_to_line_distance(point, v1, v2)
            if dist < min_distances[j]:
                min_distances[j] = dist

    return min_distances

def cost_fct(solution, data):
    x0, y0, tf, tw, bf, d = solution

    v0 = np.array([x0, y0])
    v1 = v0 + np.array([0, tf])
    v2 = v1 + np.array([(bf/2 - tw/2), 0])
    v3 = v2 + np.array([0, (d - 2*tf)])
    v5 = v0 + np.array([0, d])
    v4 = v5 - np.array([0, tf])
    v6 = v5 + np.array([bf, 0])
    v7 = v4 + np.array([bf, 0])
    v8 = v3 + np.array([tw, 0])
    v9 = v2 + np.array([tw, 0])
    v11 = v0 + np.array([bf, 0])
    v10 = v11 + np.array([0, tf])

    vertices = np.array([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11])

    # # plot lines in 2D iterate 0 - 11 and 0
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for i in range(11):
    #     ax.plot([vertices[i][0], vertices[i+1][0]], [vertices[i][1], vertices[i+1][1]])
    # ax.plot([vertices[11][0], vertices[0][0]], [vertices[11][1], vertices[0][1]])
    # ax.set_aspect('equal')
    # plt.show()

    # start = time.time()
    dists = min_distance_to_polygon(data, vertices)
    rmse = np.sqrt(np.mean(dists**2))
    # print(f'Elapsed time: {time.time() - start}')

    # # plot the lines again and add the points, color code distance to polygon
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for i in range(11):
    #     ax.plot([vertices[i][0], vertices[i+1][0]], [vertices[i][1], vertices[i+1][1]])
    # ax.plot([vertices[11][0], vertices[0][0]], [vertices[11][1], vertices[0][1]])
    # ax.scatter(data[:, 0], data[:, 1], c=dists, cmap='viridis')
    # ax.set_aspect('equal')
    # plt.show()

    return rmse

def fct(points_array_2D):
    # plot points in 2D
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(points_array_2D[:, 0], points_array_2D[:, 1], s=0.05, color='grey', zorder=7)
    ax.set_aspect('equal')


    boundings = [points_array_2D.min(axis=0), points_array_2D.max(axis=0)]
    bounding_ext_x = abs(boundings[1][0] - boundings[0][0])
    bounding_ext_y = abs(boundings[1][1] - boundings[0][1])
    # plot bounding box
    ax.plot(
        [boundings[0][0], boundings[1][0], boundings[1][0], boundings[0][0], boundings[0][0]],
        [boundings[0][1], boundings[0][1], boundings[1][1], boundings[1][1], boundings[0][1]],
        color='red',
        alpha=0.25,
        zorder=5,
        linewidth=4
    )
    plt.show()

    # initiate params and define bounds
    x0_lims = [boundings[0][0], boundings[0][0] + 0.2 * bounding_ext_x]
    y0_lims = [boundings[0][1], boundings[0][1] + 0.2 * bounding_ext_y]
    tf_lims = [0.005, 0.05]
    tw_lims = [0.005, 0.05]
    bf_lims = [0.05, bounding_ext_x]
    d_lims = [0.05, bounding_ext_y]
    lims = [x0_lims, y0_lims, tf_lims, tw_lims, bf_lims, d_lims]

    # # sample random float numbers within the defined bounds
    # x0_init = np.random.uniform(x0_lims[0], x0_lims[1])
    # y0_init = np.random.uniform(y0_lims[0], y0_lims[1])
    # tf_init = np.random.uniform(tf_lims[0], tf_lims[1])
    # tw_init = np.random.uniform(tw_lims[0], tw_lims[1])
    # bf_init = np.random.uniform(bf_lims[0], bf_lims[1])
    # d_init = np.random.uniform(d_lims[0], d_lims[1])
    #
    # solution_init = [x0_init, y0_init, tf_init, tw_init, bf_init, d_init]
    # cost_init = cost_fct(solution_init, points_array_2D)

    num_vertices = 6
    lower_bound = [lim[0] for lim in lims]
    upper_bound = [lim[1] for lim in lims]

    xopt, fopt = pso(cost_fct, lower_bound, upper_bound, args=(points_array_2D,), swarmsize=10, maxiter=10)

    optimal_vertices = xopt.reshape((num_vertices, 2))
    optimal_vertices = np.vstack([optimal_vertices, optimal_vertices[0]])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(optimal_vertices[:, 0], optimal_vertices[:, 1])
    ax.scatter(points_array_2D[:, 0], points_array_2D[:, 1], s=0.05, color='grey', zorder=7)
    ax.set_aspect('equal')
    plt.show()





    a = 0