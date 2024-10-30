import copy
import time

import matplotlib.pyplot as plt
import numpy as np


def cost_fct_1(solution_params, data_points, debug_plot=False):
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

    plot_flag_all = debug_plot
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

def params2verts_rev(solution):
    tf, tw, bf, d, x0, y0 = solution
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


def params2verts(solution, from_cog=True):
    if from_cog:
        tf, tw, bf, d = solution
        x0 = -bf / 2
        y0 = -d / 2
    else:
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


# def subdivide_edges(edges, edge_normals=None, num=None, lmax=None):
#     # subdivide long edges for increased information value of relative edge activation
#     n_to_split = np.zeros((edges.shape[0]), dtype=int)
#     new_nodes = [None for _ in range(edges.shape[0])]
#     if lmax is not None:
#         for i, edge in enumerate(edges):
#             edge_length = np.linalg.norm(edge[0] - edge[1])
#             if edge_length > lmax:
#                 # find how many subdivisions are needed
#                 n = int(np.ceil(edge_length / lmax))
#                 n_to_split[i] = n
#                 new = []
#                 for j in range(n):
#                     new.append(edge[0] + j / n * (edge[1] - edge[0]))
#                 new_nodes[i] = new
#
#
#     edges_new = np.zeros((edges.shape[0] + int(np.sum(n_to_split)), 2, 2))
#     normals_new = np.zeros_like(edges_new)
#
#     for i, n in enumerate(n_to_split):
#         if n > 0:
#             edges_new[i] = np.array([edges[i][0], new_nodes[i][0]])
#             normals_new[i] = edge_normals[i]
#             for j in range(n):
#                 edges_new[i + j + 1] = np.array([new_nodes[i][j], new_nodes[i][j + 1]])
#                 normals_new[i + j + 1] = edge_normals[i]
#             edges_new[i + n + 1] = np.array([new_nodes[i][-1], edges[i][1]])
#             normals_new[i + n + 1] = edge_normals[i]
#         else:
#             edges_new[i] = edges[i]
#             normals_new[i] = edge_normals[i]

def subdivide_edges(edges, edge_normals=None, num=None, lmax=None):
    if edge_normals is None or lmax is None:
        return edges, edge_normals

    n_splits = []
    split_points = []
    total_new_points = 0

    # Calculate splits needed
    for i, edge in enumerate(edges):
        edge_length = np.linalg.norm(edge[1] - edge[0])
        if edge_length > lmax:
            n = int(np.ceil(edge_length / lmax)) - 1
            total_new_points += n
            n_splits.append(n)
            points = []
            for j in range(1, n + 1):
                point = edge[0] + (j / (n + 1)) * (edge[1] - edge[0])
                points.append(point)
            split_points.append(points)
        else:
            n_splits.append(0)
            split_points.append([])

    # Create new arrays with correct size
    new_size = len(edges) + total_new_points
    edges_new = np.zeros((new_size, 2, 2))
    normals_new = np.zeros((new_size, 2, 2))

    # Normalize normals
    normalized_normals = edge_normals.copy()
    for i in range(len(edge_normals)):
        norm = np.linalg.norm(edge_normals[i][0])
        if norm > 0:
            normalized_normals[i] = edge_normals[i] / norm

    # Fill new arrays
    idx = 0
    for i in range(len(edges)):
        if n_splits[i] == 0:
            edges_new[idx] = edges[i]
            normals_new[idx] = normalized_normals[i]
            idx += 1
        else:
            edges_new[idx] = np.array([edges[i][0], split_points[i][0]])
            normals_new[idx] = normalized_normals[i]
            idx += 1

            for j in range(len(split_points[i]) - 1):
                edges_new[idx] = np.array([split_points[i][j], split_points[i][j + 1]])
                normals_new[idx] = normalized_normals[i]
                idx += 1

            edges_new[idx] = np.array([split_points[i][-1], edges[i][1]])
            normals_new[idx] = normalized_normals[i]
            idx += 1


    # plot edges with normals at center
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_len = 0.02
    for edge, normal in zip(edges_new, normals_new):
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]])
        center = (edge[0] + edge[1]) / 2
        ax.plot([center[0], center[0] + normal[0][0] * n_len], [center[1], center[1] + normal[0][1] * n_len])
    # equal aspect ratio
    ax.set_aspect('equal')
    plt.show()


    # subdivide edges into num segments
    if num is not None:
        raise NotImplementedError # not implemented yet

    if edge_normals is None:
        return edges_new
    else:
        return edges_new, normals_new


def get_solution_edge_normals():
    """
    Returns the edge normals of the solution polygon
    hardcoded for the I-beam
    """
    n_up = [0, 1]
    n_down = [0, -1]
    n_left = [-1, 0]
    n_right = [1, 0]
    polygon_edge_normals = np.array(
        [n_left, n_up, n_left, n_down, n_left, n_up, n_right, n_down, n_right, n_up, n_right, n_down]
    )
    return polygon_edge_normals