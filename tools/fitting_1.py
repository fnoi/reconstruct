import copy
import time

import matplotlib.pyplot as plt
import numpy as np


def cost_fct_1(solution_params, data_points):
    edge_penalty = 100
    solution_verts = params2verts(solution_params)
    solution_edges = verts2edges(solution_verts)

    dist_value = np.zeros((data_points.shape[0]))
    # dist_type = np.zeros((data_points.shape[0]))

    edge_activation = np.zeros((solution_edges.shape[0]))

    edge_tracker = [[] for _ in range(solution_edges.shape[0])]

    for i, point in enumerate(data_points):
        dist_vert = np.linalg.norm(solution_verts - point, axis=1)

        # v1v2: p2 - p1
        v1v2 = solution_edges[:, 1] - solution_edges[:, 0]
        v1p = point - solution_edges[:, 0]

        a = np.sum(v1v2 * v1p, axis=1)

        b = np.linalg.norm(v1v2, axis=1) ** 2

        t = a / b

        t_valid = np.logical_and(0 <= t, t <= 1)
        dist_edge = np.linalg.norm(solution_edges[:, 0] + t[:, None] * v1v2 - point, axis=1)
        dist_edge_valid = copy.deepcopy(dist_edge)
        dist_edge_valid[~t_valid] = 1e5
        # dist_edge[~t_valid] = 1e5

        # find min dist in vert and edge
        min_dist_vert = np.min(dist_vert)
        min_dist_vert_id = np.argmin(dist_vert)
        min_dist_edge = np.min(dist_edge_valid)
        min_dist_edge_id = np.argmin(dist_edge_valid)

        if min_dist_edge < min_dist_vert:
            dist_value[i] = min_dist_edge
            # dist_type[i] = 3
            edge_activation[min_dist_edge_id] += 1
            edge_tracker[min_dist_edge_id].append(i)
        else:
            # dist_value[i] = min_dist_vert
            # dist_type[i] = 1
            # active_vert = np.argmin(dist_vert)
            active_edges = get_neighboring_edges_from_vert(min_dist_vert_id, solution_verts.shape[0])
            # find the edge that is closer to the point


            edge_activation[np.asarray(active_edges)] += 1
            edge_tracker[active_edges[0]].append(i)
            edge_tracker[active_edges[1]].append(i)

        plot_flag_single = False
        if plot_flag_single:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for edge in solution_edges:
                ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]])
            ax.plot(point[0], point[1], 'ro')
            plt.show()

    plot_flag_all = False
    if plot_flag_all:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        edge_colors = list(edge_activation / np.max(edge_activation))
        # retrieve viridis color for each edge
        edge_colors_viridis = plt.cm.spring(edge_colors)

        i = 0
        for edge, edge_color in zip(solution_edges, edge_colors_viridis):
            ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], c=edge_color)
            # plot value as text next to line
            ax.text((edge[0][0] + edge[1][0]) / 2, (edge[0][1] + edge[1][1]) / 2, f'{int(edge_activation[i])}',
                    fontsize=10, color='black')
            i += 1
        ax.scatter(data_points[:, 0], data_points[:, 1], s=3, c=list(dist_value / np.max(dist_value)))
        ax.set_aspect('equal')
        plt.show()

    for _i in range(3):
        for i, edge_count in enumerate(edge_activation):
            if edge_count == 0:
                neighboring_edges_ids = (
                    (i - 1) % solution_edges.shape[0],
                    (i + 1) % solution_edges.shape[0]
                )
                # neighbor_min_dist =
                if edge_activation[neighboring_edges_ids[0]] > 0 or edge_activation[neighboring_edges_ids[1]] > 0:
                    if edge_activation[neighboring_edges_ids[0]] > edge_activation[neighboring_edges_ids[1]]:
                        # case 1: neighor edge 0 has more activation
                        a = 1
                    else:
                        a = 0


    for i, edge_count in enumerate(edge_activation):
        if edge_count != 0:
            weight = 1 / (edge_count / len(data_points) + 0.01)
            a = 0
        else:
            weight = 1
            neighboring_edge_ids = (
                (i - 1) % solution_edges.shape[0],
                (i + 1) % solution_edges.shape[0]
            )

    for i, edge_count in enumerate(edge_activation):
        if edge_count == 0:
            weight = 1
            neighboring_edge_ids = (
                (i - 1) % solution_edges.shape[0],
                (i + 1) % solution_edges.shape[0]
            )

            a = 0

    a = 0

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


def get_neighboring_edges_from_vert(vertex_index, num_edges):
    prev_edge_index = (vertex_index - 1) % num_edges
    next_edge_index = vertex_index % num_edges
    return prev_edge_index, next_edge_index


def get_neighboring_edges_from_edge(edge_index, num_edges):
    prev_vert_index = edge_index
    next_vert_index = (edge_index + 1) % num_edges
    return int(prev_vert_index), int(next_vert_index)
