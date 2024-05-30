import time

import numpy as np

from tools.fitting_0 import param2vertices, my_dist2edge


def verts2edges(vertices):
    num_edges = vertices.shape[0]
    edges = np.zeros((num_edges, 2, 2))
    for i in range(num_edges):
        edges[i] = np.array([vertices[i], vertices[(i + 1) % num_edges]])
    return edges


def dist_verts_edges(vertices, edges, data_points):
    # for each point calc all vert distances and all edge distances, pick the smallest, store the type
    dist2verts = np.zeros((data_points.shape[0], vertices.shape[0]))
    #TODO: pick up here


def cost_fct_1(solution_params, data_points):
    solution_verts = param2vertices(solution_params)
    solution_edges = verts2edges(solution_verts)

    timer = time.time()
    dist, type = my_dist2edge(solution_edges, data_points)
    print(f'elapsed time: {time.time() - timer}')

    timer = time.time()
    dist, type = dist_verts_edges(solution_verts, solution_edges, data_points)
    print(f'elapsed time: {time.time() - timer}')


    # setup edges
    # calc distance to edges
    # evaluate wether closer to edge or vertex defining edge
    # calculate edge activity

    a = 0
