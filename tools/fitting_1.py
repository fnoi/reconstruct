import copy
import time

import matplotlib.pyplot as plt
import numpy as np


def cost_fct_1(solution_params, data_points):
    # edge_penalty: penalty for inactive edges
    edge_penalty = 100
    # solution_verts: vertices of the cs-polygon defined by the solution parameters
    solution_verts = params2verts(solution_params)
    # solution_edges: edges of the cs-polygon defined by the solution parameters
    solution_edges = verts2edges(solution_verts)

    # edge_activation: number of points that are closest to each edge
    edge_activation = np.zeros((solution_edges.shape[0]))
    # edge_track_ids: list of point ids that are closest to each edge
    edge_track_ids = [[] for _ in range(solution_edges.shape[0])]
    # edge_track_dists: list of distances of points that are closest to each edge
    edge_track_dists = [[] for _ in range(solution_edges.shape[0])]
    # point_best: best distance to the polygon for each point   # remove to improve performance
    point_best = np.zeros((data_points.shape[0]))

    for i, point in enumerate(data_points):
        dist_vert = np.linalg.norm(solution_verts - point, axis=1)

        # distance line to polygon: vertices, lines, ...
        # https://stackoverflow.com/questions/10983872/distance-from-a-point-to-a-polygon
        v1v2 = solution_edges[:, 1] - solution_edges[:, 0]
        v1p = point - solution_edges[:, 0]
        t = np.sum(v1v2 * v1p, axis=1) / np.linalg.norm(v1v2, axis=1) ** 2
        t_valid = np.logical_and(0 <= t, t <= 1)

        dist_edge = np.linalg.norm(solution_edges[:, 0] + t[:, None] * v1v2 - point, axis=1)
        dist_edge_valid = copy.deepcopy(dist_edge)
        dist_edge_valid[~t_valid] = 1e5

        min_dist_vert = np.min(dist_vert)
        min_dist_vert_id = np.argmin(dist_vert)
        min_dist_edge = np.min(dist_edge_valid)
        min_dist_edge_id = np.argmin(dist_edge_valid)

        if min_dist_edge < min_dist_vert:
            edge_activation[min_dist_edge_id] += 1
            edge_track_ids[min_dist_edge_id].append(i)
            edge_track_dists[min_dist_edge_id].append(min_dist_edge)
            point_best[i] = min_dist_edge  # remove to improve performance
        else:
            active_edges = get_neighboring_edges_from_vert(min_dist_vert_id, solution_verts.shape[0])
            edge_activation[np.asarray(active_edges)] += 1
            # find edge distances for both edges, do not track ids!
            edge_track_dists[active_edges[0]].append(dist_edge[active_edges[0]])
            edge_track_dists[active_edges[1]].append(dist_edge[active_edges[1]])
            point_best[i] = min_dist_vert  # remove to improve performance

    plot_flag_all = False
    if plot_flag_all:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        edge_colors = list(edge_activation / np.max(edge_activation))
        edge_colors_viridis = plt.cm.viridis(edge_colors)

        for i, (edge, edge_color) in enumerate(zip(solution_edges, edge_colors_viridis)):
            ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], c=edge_color)
            ax.text((edge[0][0] + edge[1][0]) / 2, (edge[0][1] + edge[1][1]) / 2, f'{int(edge_activation[i])}',
                    fontsize=10, color='black')
        ax.scatter(data_points[:, 0], data_points[:, 1], s=3, c=list(point_best / np.max(point_best)))
        ax.set_aspect('equal')
        plt.show()

    cost = np.zeros_like(edge_activation)
    for i, edge in enumerate(solution_edges):
        edge_length = np.linalg.norm(edge[0] - edge[1])
        if edge_activation[i] == 0:
            weight = 1
            neighbor_edges = (
                (i - 1) % solution_edges.shape[0],
                (i + 1) % solution_edges.shape[0]
            )
            neighbor_0 = edge_track_ids[neighbor_edges[0]]
            neighbor_1 = edge_track_ids[neighbor_edges[1]]
            if len(neighbor_0) == 0 and len(neighbor_1) == 0:  # cannot be activated by neighbors
                cost[i] = edge_penalty * edge_length
            else:  # activated by neighbor
                neighbor_ids = np.unique(np.concatenate((neighbor_0, neighbor_1)))

                line_dir = edge[0] - edge[1]
                point_to_line = data_points[neighbor_ids.astype(int), :] - edge[0]
                t = np.dot(point_to_line, line_dir) / np.dot(line_dir, line_dir)

                edge_to_points = np.linalg.norm(point_to_line - t[:, None] * line_dir, axis=1)
                best_dist = np.min(edge_to_points)
                cost[i] = weight * best_dist * edge_length
        else:
            weight = 1 / (len(edge_track_ids[i]) / len(data_points) + 0.01)
            cost[i] = weight * np.sum(edge_track_dists[i]) * edge_length  # corresponds to MAE of edge

    return np.sum(cost)


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
