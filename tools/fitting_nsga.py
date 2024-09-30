import copy
import pickle
import random
from typing import final

import deap
import numpy as np
from deap import creator, base, tools, algorithms

from functools import partial

from matplotlib import pyplot as plt

from tools.fitting_1 import params2verts, params2verts_rev, verts2edges
from tools.fitting_pso import cs_plot


def setup_lims_placement(points):
    boundings = [points.min(axis=0), points.max(axis=0)]

    extent_x = abs(boundings[1][0] - boundings[0][0])
    extent_y = abs(boundings[1][1] - boundings[0][1])

    # initiate params and define bounds
    rel_ext = 0.1

    limits_x_moved = (boundings[0][0] - rel_ext * extent_x, boundings[1][0])
    limits_y_moved = (boundings[0][1] - rel_ext * extent_y, boundings[1][1])

    return limits_x_moved, limits_y_moved


def solve_w_nsga(points):
    with open('/Users/fnoic/PycharmProjects/reconstruct/data/beams/beams_frame.pkl', 'rb') as f:
        data = pickle.load(f)
    data = data.dropna()

    # setup complete input data for solution finding
    parameter_set = data[['tw', 'tf', 'bf', 'd']].values.tolist()
    x_range, y_range = setup_lims_placement(points)

    points_array = copy.deepcopy(points)
    points = [row.tolist() for row in points]

    ###########
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Register the function to select a random row from the parameter set
    toolbox.register("attr_row", random.choice, parameter_set)
    toolbox.register("attr_x_offset", random.uniform, *x_range)
    toolbox.register("attr_y_offset", random.uniform, *y_range)

    # Define how to form an individual
    def create_individual():
        row = toolbox.attr_row()
        x_offset = toolbox.attr_x_offset()
        y_offset = toolbox.attr_y_offset()
        return creator.Individual(row + [x_offset, y_offset])

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fast_fitness, point_cloud=points)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutate, indpb=0.2, parameter_set=parameter_set, x_range=x_range, y_range=y_range)
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=400)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    final_pop, log = algorithms.eaMuPlusLambda(
        population, toolbox,
        mu=50, lambda_=400, cxpb=0.5, mutpb=0.2,
        ngen=10, stats=stats, halloffame=hof, verbose=True
    )

    # Save final population
    with open('final_population.txt', 'w') as f:
        for ind in final_pop:
            f.write(str(ind) + '\n')

    # Save log statistics to a separate file
    with open('ga_log.txt', 'w') as f:
        for record in log:
            f.write(str(record) + '\n')

    # Extract the Pareto front
    pareto_front = tools.sortNondominated(final_pop, len(final_pop), first_front_only=True)

    # Save the Pareto front to a text file
    with open('pareto_front.txt', 'w') as f:
        for ind in pareto_front[0]:  # pareto_front[0] contains the first front, the actual Pareto front
            f.write(str(ind) + " Fitness: " + str(ind.fitness.values) + '\n')

    final_params = final_pop[0]

    final_verts = params2verts_rev(final_params)

    cs_plot(final_verts, points_array)
    raise ValueError("This is a test exception")

###########

def distance_to_lines(points, line):
    # Extract coordinates of the line segment
    x1, y1 = line["start"]
    x2, y2 = line["end"]

    # Calculate the components for the line equation
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - y2 * x1

    # Calculate the perpendicular distance from the point to the line
    point_distance = np.abs(A * points[0] + B * points[1] + C) / np.sqrt(A ** 2 + B ** 2)

    # Check if the point is within the extents of the line segment (within threshold)
    # Parametric representation of the line segment to check the projection of the point
    dot_product = (points[0] - x1) * (x2 - x1) + (points[1] - y1) * (y2 - y1)
    squared_length = (x2 - x1) ** 2 + (y2 - y1) ** 2

    # Calculate the parameter t which represents where the projection of the point lies
    t = dot_product / squared_length

    # If the projection of the point lies between 0 and 1, it falls within the segment's extents
    if t < 0 or t > 1:
        # If the projected point is outside the segment, return a large value
        return 100
    else:
        # Return the actual distance if within extents and within threshold
        return point_distance

def distance_to_edge(point, edge):
    return distance_to_lines(point, {"start": edge[0], "end": edge[1]})

def fitness_function_2d_2(individual, point_cloud):
    web_thickness, flange_thickness, width, height, x_offset, y_offset = individual
    verts = params2verts_rev(individual)
    edges = verts2edges(verts)

    count_within_threshold = 0
    aggregated_distance = 0

    threshold = 0.002
    for point in point_cloud:
        # Check if the point is within the threshold distance for any plane
        # if any(distance_to_edge(point, edge) <= threshold for edge in edges):
        #     count_within_threshold += 1
        aggregated_distance += min(distance_to_edge(point, edge) ** 2 for edge in edges)

    return aggregated_distance,

    # return -count_within_threshold,


def fast_fitness(individual, point_cloud):
    """efficient distance computation, goal is to enable eval without prior down sampling"""
    n_points = len(point_cloud)
    n_vertices = 12 # hardcode for h-beam

    verts = params2verts_rev(individual)
    edge_starts = verts
    edge_ends = np.roll(verts, -1, axis=0)

    distances = np.inf * np.ones(n_points)

    for start, end in zip(edge_starts, edge_ends):
        edge_dist = point_segment_distance(point_cloud, start, end)
        distances = np.minimum(distances, edge_dist)

    return np.sum(distances),



def point_segment_distance(points, edge_start, edge_end):
    edge_vec = edge_end - edge_start
    edge_len_sq = np.sum(edge_vec ** 2)

    if edge_len_sq == 0:
        return np.linalg.norm(points - edge_start, axis=1)

    t = np.sum((points - edge_start) * edge_vec, axis=1) / edge_len_sq
    t = np.clip(t, 0, 1)

    projection = edge_start + t[:, None] * edge_vec

    return np.linalg.norm(points - projection, axis=1)



    a = 0







def custom_mutate(individual, indpb, parameter_set, x_range, y_range):
    # Mutate the part from parameter set
    if random.random() < indpb:
        new_values = random.choice(parameter_set)
        individual[:4] = new_values  # Assuming the first 4 are from parameter set

    # Mutate x_offset
    if random.random() < indpb:
        individual[4] = random.uniform(*x_range)

    # Mutate y_offset
    if random.random() < indpb:
        individual[5] = random.uniform(*y_range)

    return individual,

if __name__ == "__main__":
    dummy_individual = [0.02, 0.02, 0.2, 0.2, 0.5, 0.5]
    with open('/Users/fnoic/Library/CloudStorage/OneDrive-TUM/temp/beam_2_dump_2D.txt', 'rb') as f:
        data = np.loadtxt(f)
    fitness = fitness_function_2d_2(dummy_individual, data)

    a = 0