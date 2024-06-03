import time

import matplotlib.pyplot as plt
import numpy as np


def dist_verts_edges(vertices, edges, data_points):
    dist_value = np.zeros((data_points.shape[0]))
    dist_type = np.zeros((data_points.shape[0]))

    activation_edges = np.zeros((data_points.shape[0]))

    # for each point calc all vert distances and all edge distances, pick the smallest, store the type
    dists_verts = np.zeros((data_points.shape[0], vertices.shape[0]))
    dist_edges = np.zeros((data_points.shape[0], edges.shape[0]))
    dists = np.zeros((data_points.shape[0]))  # 0 non determined
    type = np.zeros((data_points.shape[0]))  # 0 empty, non determined 1 vertex 2 edge

    for i, point in enumerate(data_points):
        dist_vert = np.linalg.norm(vertices - point, axis=1)
        # dists_verts[i] = dist_vert

        # v1v2: p2 - p1
        v1v2 = edges[:, 1] - edges[:, 0]
        v1p = point - edges[:, 0]

        a = np.sum(v1v2 * v1p, axis=1)

        b = np.linalg.norm(v1v2, axis=1) ** 2

        t = a / b

        t_valid = np.logical_and(0 <= t, t <= 1)
        dist_edge = np.linalg.norm(edges[:, 0] + t[:, None] * v1v2 - point, axis=1)
        dist_edge[~t_valid] = 1e5

        # find min dist in vert and edge
        min_dist_vert = np.min(dist_vert)
        min_dist_edge = np.min(dist_edge)
        if min_dist_edge < min_dist_vert:
            dist_value[i] = min_dist_edge
            dist_type[i] = 3
            active_edge = np.argmin(dist_edge)
            activation_edges[active_edge] += 1
        else:
            dist_value[i] = min_dist_vert
            dist_type[i] = 1
            active_vert = np.argmin(dist_vert)
            active_edges = get_neighboring_edges(active_vert, vertices.shape[0])
            activation_edges[np.asarray(active_edges)] += 1

        plot_flag = False
        if plot_flag:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for edge in edges:
                ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]])
            ax.plot(point[0], point[1], 'ro')
            plt.show()

    return dist_value, dist_type, activation_edges


def cost_fct_1(solution_params, data_points):
    edge_penalty = 100
    solution_verts = params2verts(solution_params)
    solution_edges = verts2edges(solution_verts)

    dists, types, active_edges = dist_verts_edges(solution_verts, solution_edges, data_points)

    # normalize active edges
    weights = active_edges / np.sum(active_edges)
    # penalize inactive edges
    penalty = np.sum(active_edges == 0) * edge_penalty

    cost = np.sum(dists) + penalty

    return cost


def params2verts(solution):
    x0, y0, tf, tw, bf, d = solution

    v0 = np.array([x0, y0])
    v1 = v0 + np.array([0, tf])
    v2 = v1 + np.array([(bf / 2 - tw / 2), 0])
    v3 = v2 + np.array([0, (d - 2 * tf)])
    v5 = v0 + np.array([0, d])
    v4 = v5 - np.array([0, tf])
    v6 = v5 + np.array([bf, 0])
    v7 = v4 + np.array([bf, 0])
    v8 = v3 + np.array([tw, 0])
    v9 = v2 + np.array([tw, 0])
    v11 = v0 + np.array([bf, 0])
    v10 = v11 + np.array([0, tf])

    vertices = np.array([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11])

    return vertices


def verts2edges(vertices):
    num_edges = vertices.shape[0]
    edges = np.zeros((num_edges, 2, 2))
    for i in range(num_edges):
        edges[i] = np.array([vertices[i], vertices[(i + 1) % num_edges]])
    return edges


def get_neighboring_edges(vertex_index, num_edges):
    prev_edge_index = (vertex_index - 1) % num_edges
    next_edge_index = vertex_index % num_edges
    return prev_edge_index, next_edge_index
