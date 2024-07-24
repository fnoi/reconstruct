import time
import sys

import numpy as np
from matplotlib import pyplot as plt

from pyswarm import pso

from tools.fitting_0 import cost_fct_0
from tools.fitting_1 import cost_fct_1, params2verts


def point_to_line_distance(point, v1, v2):
    """Calculate the minimum distance from a point to a line segment defined by vertices v1 and v2."""
    line_vec = np.array(v2) - np.array(v1)
    point_vec = np.array(point) - np.array(v1)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    t = np.clip(t, 0, 1)
    nearest = np.array(v1) + t * line_vec
    dist = np.linalg.norm(nearest - np.array(point))
    return dist


def min_distance_to_polygon(points, vertices, active_edges=False):
    """Calculate the minimum distance from each point in 'points' to a polygon defined by 'vertices'.
       Optionally return the number of polygon edges that are not the closest to any point."""
    num_vertices = len(vertices)
    num_points = len(points)
    min_distances = np.inf * np.ones(num_points)
    edge_closest_count = np.zeros(num_vertices, dtype=int)  # Array to count closest occurrences for each edge

    for i in range(num_vertices):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % num_vertices]  # Wrap around to connect the last vertex to the first
        for j in range(num_points):
            point = points[j, :2]  # Assuming points are in nx3, ignore the third dimension if present
            dist = point_to_line_distance(point, v1, v2)
            if dist < min_distances[j]:
                min_distances[j] = dist
                if active_edges:
                    edge_closest_count[i] += 1  # Mark this edge as closest for this point

    if active_edges:
        # Count how many edges were never the closest
        inactive_edges_count = np.sum(edge_closest_count == 0)
        return min_distances, inactive_edges_count
    else:
        return min_distances


def cs_plot(vertices=None, points=None):
    # plot lines in 2D iterate 0 - 11 and 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if vertices is not None:
        for i in range(11):
            ax.plot([vertices[i][0], vertices[i + 1][0]], [vertices[i][1], vertices[i + 1][1]])
        ax.plot([vertices[11][0], vertices[0][0]], [vertices[11][1], vertices[0][1]])
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], s=0.05, color='grey')
    ax.set_aspect('equal')
    plt.show()


def plot_2D_points_bbox(points_array_2D):
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

def fitting_fct(points_array_2D):
    boundings = [points_array_2D.min(axis=0), points_array_2D.max(axis=0)]
    bounding_ext_x = abs(boundings[1][0] - boundings[0][0])
    bounding_ext_y = abs(boundings[1][1] - boundings[0][1])

    # initiate params and define bounds #TODO: consider to initiate from table values from the standard
    rel_ext = 0.1
    x0_lims = [boundings[0][0] - rel_ext * bounding_ext_x, boundings[0][0] + rel_ext * bounding_ext_x]
    y0_lims = [boundings[0][1] - rel_ext * bounding_ext_y, boundings[0][1] + rel_ext * bounding_ext_y]
    tf_lims = [0.005, 0.02]
    tw_lims = [0.005, 0.02]
    bf_lims = [0.1, bounding_ext_x]
    d_lims = [0.1, bounding_ext_y]
    lims = [x0_lims, y0_lims, tf_lims, tw_lims, bf_lims, d_lims]

    num_vertices = 6
    lower_bound = [lim[0] for lim in lims]
    upper_bound = [lim[1] for lim in lims]

    # x0 = np.min(points_array_2D[:, 0])
    # y0 = np.min(points_array_2D[:, 1])
    # dummy_solution = np.array([x0, y0, 0.01, 0.01, 0.1, 0.1])
    # timer = time.time()
    # cost_0 = cost_fct_0(dummy_solution, points_array_2D)
    # elapsed_0 = time.time() - timer
    # print(f'elapsed time: {elapsed_0:.3f}')
    # timer = time.time()
    # cost_1 = cost_fct_1(dummy_solution, points_array_2D)
    # elapsed_1 = time.time() - timer
    # print(f'elapsed time: {elapsed_1:.3f}')
    # rel_improvement = 1 - elapsed_1 / elapsed_0
    # print(f'v1/v0: {elapsed_1/elapsed_0:.3f}; relative improvement: {rel_improvement:.3f}')

    swarm_size = 200
    maxiter = 100

    # omega =
    phip = 2.05
    phig = 2.05
    # minstep =
    # minfunc =


    timee = time.time()

    # ieqcons=[], f_ieqcons=None, args=(), kwargs={},
    #         swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
    #         minstep=1e-8, minfunc=1e-8, debug=False)

    timestamp_very_precise = time.time()
    original_stdout = sys.stdout
    with open(f'/Users/fnoic/PycharmProjects/reconstruct/experiment_log/output_{timestamp_very_precise}.txt', 'w') as f:
        sys.stdout = f

        # start writing output to file
        xopt, fopt = pso(func=cost_fct_1,
                         lb=lower_bound,
                         ub=upper_bound,
                         args=(points_array_2D,),
                         swarmsize=swarm_size,
                         maxiter=maxiter,
                         phip=phip,
                         phig=phig,
                         debug=True)
        sys.stdout = original_stdout


    # xopt, fopt = pso(cost_fct_1, lower_bound, upper_bound, args=(points_array_2D,),
    #                  swarmsize=swarm_size, maxiter=max_iter)
    print(f'pso time to complete {time.time() - timee:2f}')

    optimal_vertices = params2verts(xopt)
    cs_plot(optimal_vertices, points_array_2D)

    return xopt, optimal_vertices, fopt
