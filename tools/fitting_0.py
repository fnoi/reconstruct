import numpy as np
from matplotlib import pyplot as plt


# Placeholder definitions for required utility functions
def edge2vert(edge, num_edges):
    if edge == num_edges - 1:  # Adjusting index for 0-based in Python
        v1 = edge
        v2 = 0  # Wraps around to the first vertex for the last edge
    else:
        v1 = edge
        v2 = edge + 1
    return [v1, v2]



def vert2edge(vert, num_verts):
    if vert == 1:
        e1 = num_verts - 1  # Last edge, adjusted for zero-based indexing
        e2 = 0  # First edge in zero-based index
    else:
        e1 = vert - 2  # Subtracting 2: one for zero-based index and one to get the previous edge
        e2 = vert - 1  # Zero-based index for the current vertex's direct edge
    return [e1, e2]



def my_dist2edge(v1, v2, point):
    v1v2 = np.array(v2) - np.array(v1)
    v1p = np.array(point) - np.array(v1)
    t = np.dot(v1v2, v1p) / np.linalg.norm(v1v2)**2

    if t < 0:
        dist = np.linalg.norm(v1p)
        index = 0  # Closest to vertex v1
    elif t > 1:
        dist = np.linalg.norm(point - np.array(v2))
        index = 1  # Closest to vertex v2
    else:
        projection = np.array(v1) + t * v1v2
        dist = np.linalg.norm(projection - np.array(point))
        index = 2  # Closest to somewhere along the edge

    return dist, index

def cost_fct_0(solution, data):
    vertices = params2verts(solution)
    num_edges = vertices.shape[0]
    num_points = data.shape[0]
    edge_pts = [[] for _ in range(num_edges)]
    penalty = 100
    activity_dist = np.zeros(num_edges)
    edge_dist = np.zeros(num_edges)

    for i, point in enumerate(data):
        best_dist = np.inf
        edge_flag = False
        best_index = -1

        for j in range(num_edges):
            verts_id = edge2vert(j, num_edges)
            v1_id, v2_id = verts_id[0], verts_id[1]
            v1, v2 = vertices[v1_id], vertices[v2_id]
            dist, index = my_dist2edge(v1, v2, point)

            if dist < best_dist:
                best_dist = dist
                if index == 0:
                    best_index = v1_id
                elif index == 1:
                    best_index = v2_id
                else:
                    best_index = j
                    edge_flag = True

        if edge_flag:
            edge_dist[best_index] += best_dist
            edge_pts[best_index].append(i)
        else:
            edges = vert2edge(best_index, num_edges)
            edge_dist[edges[0]] += 0.7071 * best_dist
            edge_dist[edges[1]] += 0.7071 * best_dist

    # Activity and edge distance calculations
    edge_points = np.zeros(num_edges)
    for j in range(num_edges):
        num_edge_points = edge_pts[j].count(j)
        edge_points[j] = num_edge_points

    # Weights and final calculation
    weights = edge_points / num_points
    edge_weights = 1 / (weights + 0.02)
    final_edge_dist = edge_dist + activity_dist

    z = np.mean(edge_weights * final_edge_dist)

    return z


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
