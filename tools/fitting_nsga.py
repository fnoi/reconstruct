import copy
import pickle
import random
from typing import final

import deap
import numpy as np
import plotly.graph_objects as go
from deap import creator, base, tools, algorithms

from functools import partial

from matplotlib import pyplot as plt

from scipy.interpolate import splprep, splev

from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist

from tools.fitting_1 import params2verts, params2verts_rev, verts2edges
from tools.fitting_pso import cs_plot


def setup_lims_placement(points):
    boundings = [points.min(axis=0), points.max(axis=0)]

    extent_x = abs(boundings[1][0] - boundings[0][0])
    extent_y = abs(boundings[1][1] - boundings[0][1])

    # initiate params and define bounds
    rel_ext = 0.1
    rel_int = 0.5

    limits_x_moved = (boundings[0][0] - rel_ext * extent_x, boundings[0][0] + rel_int * extent_x)
    limits_y_moved = (boundings[0][1] - rel_ext * extent_y, boundings[0][1] + rel_int * extent_y)

    return limits_x_moved, limits_y_moved


def solve_w_nsga(points, config):
    with open('/Users/fnoic/PycharmProjects/reconstruct/data/beams/beams_frame.pkl', 'rb') as f:
        data = pickle.load(f)
    data = data.dropna()
    # load config


    # setup complete input data for solution finding // sequence is tw, tf, bf, d
    parameter_set = data[['tw', 'tf', 'bf', 'd']].values.tolist()
    x_range, y_range = setup_lims_placement(points)

    points_array = copy.deepcopy(points)
    # points_array = np.asarray(points_array)
    points = [row.tolist() for row in points]

    ###########
    # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))                 # log_distance, active_edge_length_relative, activation_distance
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0)) # minimize log distance, maximize relative active edge length, minimize edge activation distance
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    cs_ids = [_ for _ in range(len(data))]
    # Register the function to select a random row from the parameter set
    toolbox.register("attr_row", random.randint, 0, len(data) - 1)
    toolbox.register("attr_x_offset", random.uniform, *x_range)
    toolbox.register("attr_y_offset", random.uniform, *y_range)

    # Define how to form an individual
    def create_individual():
        row = toolbox.attr_row()
        x_offset = toolbox.attr_x_offset()
        y_offset = toolbox.attr_y_offset()
        return creator.Individual([row, x_offset, y_offset])

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", cost_combined, data_points=points, data_frame=data)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutate, indpb=0.2, parameter_set=parameter_set, x_range=x_range, y_range=y_range)
    # toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=config.cs_fit.n_pop)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    final_pop, log, all_individuals = eaMuPlusLambda_history(
        population, toolbox,
        mu=config.cs_fit.n_mu, lambda_=config.cs_fit.n_lambda, cxpb=0.5, mutpb=0.2,
        ngen=config.cs_fit.n_gen, stats=stats, halloffame=hof, verbose=True
    )

    # Save final population
    with open('final_population.txt', 'w') as f:
        for ind in final_pop:
            f.write(str(ind) + '\n')

    # Save log statistics to a separate file
    with open('ga_log.txt', 'w') as f:
        for record in log:
            f.write(str(record) + '\n')

    # Extract the Pareto front # final pop or all??
    pareto_front = tools.sortNondominated(all_individuals, len(all_individuals), first_front_only=True)

    # Save the Pareto front to a text file
    with open('pareto_front.txt', 'w') as f:
        for ind in pareto_front[0]:  # pareto_front[0] contains the first front, the actual Pareto front
            f.write(str(ind) + " Fitness: " + str(ind.fitness.values) + '\n')

    final_params = final_pop[0]
    final_params = final_params[1:] + list(data.iloc[final_params[0]][['tw', 'tf', 'bf', 'd']].values)

    final_verts = params2verts(final_params)
    # final_verts = params2verts_rev(final_params)

    cs_plot(final_verts, points_array)

    pareto_plot = True
    if pareto_plot:
        fig = plot_all_generations_hof_and_pareto_front(all_individuals,
                                                        hof,
                                                        pareto_front,
                                                        ["Log Distance", "Active Edge Length", "Activation Distance"],
                                                        plot_pareto_surface=False
                                                        )
        fig.show()

    # delete Individual and FitnessMin
    del creator.FitnessMulti
    del creator.Individual

    # should return h_beam_params, h_beam_verts and final cost
    h_beam_params = final_params
    h_beam_verts = final_verts
    h_beam_cost = final_pop[0].fitness.values[0]

    raise ValueError("This is a test exception")

    return h_beam_params, h_beam_verts, h_beam_cost

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

def fitness_function_2d_2(individual, data_points, data_frame):
    # web_thickness, flange_thickness, width, height, x_offset, y_offset = individual
    row_id, x_offset, y_offset = individual
    params = data_frame.iloc[row_id]
    params = params[['tw', 'tf', 'bf', 'd']].values.tolist()
    individual = [x_offset, y_offset] + params
    verts = params2verts(individual)
    # verts = params2verts_rev(individual)
    edges = verts2edges(verts)

    count_within_threshold = 0
    aggregated_distance = 0

    threshold = 0.002
    for point in data_points:
        # Check if the point is within the threshold distance for any plane
        # if any(distance_to_edge(point, edge) <= threshold for edge in edges):
        #     count_within_threshold += 1
        aggregated_distance += min(distance_to_edge(point, edge) ** 2 for edge in edges)

    return aggregated_distance,

    # return -count_within_threshold,


def point_segment_distance(points, edge_start, edge_end):
    edge_vec = edge_end - edge_start
    edge_len_sq = np.sum(edge_vec ** 2)

    if edge_len_sq == 0:
        return np.linalg.norm(points - edge_start, axis=1)

    t = np.sum((points - edge_start) * edge_vec, axis=1) / edge_len_sq
    t = np.clip(t, 0, 1)

    projection = edge_start + t[:, None] * edge_vec

    return np.linalg.norm(points - projection, axis=1)


def cost_log_distance(solution_params, data_points, data_frame):
    data_points = np.array(data_points)

    params = data_frame.iloc[solution_params[0]]
    params = params[['tw', 'tf', 'bf', 'd']].values.tolist()
    solution_params = [solution_params[1], solution_params[2]] + params
    solution_verts = params2verts(solution_params)
    solution_edges = verts2edges(solution_verts)

    edge_distances = np.array([point_segment_distance(data_points, edge[0], edge[1])
                               for edge in solution_edges])

    min_distances = np.min(edge_distances, axis=0)

    return np.sum(np.log(min_distances)),


def cost_combined(solution_params, data_points, data_frame):
    data_points = np.array(data_points)

    params = data_frame.iloc[solution_params[0]]
    params = params[['tw', 'tf', 'bf', 'd']].values.tolist()
    solution_params = [solution_params[1], solution_params[2]] + params
    solution_verts = params2verts(solution_params)
    solution_edges = verts2edges(solution_verts)

    edge_lengths = np.linalg.norm(solution_edges[:, 1] - solution_edges[:, 0], axis=1)
    edge_length_total = np.sum(edge_lengths)

    edge_distances = np.array([point_segment_distance(data_points, edge[0], edge[1])
                                 for edge in solution_edges])

    best_edge_per_point = np.argmin(edge_distances, axis=0)
    min_distances_per_point = np.min(edge_distances, axis=0)
    log_distance = np.sum(np.log(min_distances_per_point)) / len(data_points)

    edge_activity = np.zeros(len(solution_edges))
    edge_activation_dist = np.zeros(len(solution_edges))

    for edge in range(len(solution_edges)):
        if edge in best_edge_per_point:
            edge_activity[edge] = 1
    edge_activity_log = copy.deepcopy(edge_activity)

    for edge in range(len(solution_edges)):
        if edge_activity_log[edge] == 1:
            neighbor_low = edge - 1 if edge - 1 >= 0 else len(solution_edges) - 1
            neighbor_high = edge + 1 if edge + 1 < len(solution_edges) else 0
            active_low = edge_activity_log[neighbor_low] == 0
            active_high = edge_activity_log[neighbor_high] == 0
            if active_low or active_high:
                edge_activity[edge] = 0
                if active_low:
                    dist_low = np.min(edge_distances[neighbor_low])
                else:
                    dist_low = np.inf
                if active_high:
                    dist_high = np.min(edge_distances[neighbor_high])
                else:
                    dist_high = np.inf

                if dist_low < dist_high:
                    edge_activation_dist[edge] = dist_low
                else:
                    edge_activation_dist[edge] = dist_high
            else:
                continue

    active_edge_length_relative = np.sum(edge_activity * edge_lengths) / edge_length_total
    activation_distance = np.sum(edge_activation_dist)

    return log_distance, active_edge_length_relative, activation_distance



def efficient_cost_function(solution_params, data_points, data_frame, edge_penalty=100):
    """
    Updated efficient cost function to match cost_fct_1 behavior.

    :param solution_params: Parameters defining the polygon solution
    :param data_points: np.array of shape (n, 2) containing the data points
    :param data_frame: DataFrame containing the parameters
    :param edge_penalty: Penalty for inactive edges
    :return: Total cost
    """
    data_points = np.array(data_points)

    # Convert solution parameters to vertices and edges
    params = data_frame.iloc[solution_params[0]]
    params = params[['tw', 'tf', 'bf', 'd']].values.tolist()
    solution_params = [solution_params[1], solution_params[2]] + params
    solution_verts = params2verts(solution_params)
    # cs_plot(solution_verts, data_points)
    solution_edges = verts2edges(solution_verts)

    # Calculate distances from all points to all vertices
    vert_distances = cdist(data_points, solution_verts)

    # Calculate distances from all points to all edges
    edge_distances = np.array([point_segment_distance(data_points, edge[0], edge[1])
                               for edge in solution_edges])

    # Find the closest feature (edge or vertex) for each point
    min_vert_distances = np.min(vert_distances, axis=1)
    min_edge_distances = np.min(edge_distances, axis=0)

    is_closest_vert = min_vert_distances < min_edge_distances
    closest_vert_indices = np.argmin(vert_distances, axis=1)
    closest_edge_indices = np.argmin(edge_distances, axis=0)

    # Initialize edge activation and tracking
    edge_activation = np.zeros(len(solution_edges))
    edge_track_dists = [[] for _ in range(len(solution_edges))]

    # Process each point
    for i, (is_vert, vert_idx, edge_idx) in enumerate(zip(is_closest_vert, closest_vert_indices, closest_edge_indices)):
        if is_vert:
            # Activate neighboring edges for vertex points
            active_edges = get_neighboring_edges_from_vert(vert_idx, len(solution_verts))
            edge_activation[active_edges] += 1
            for ae in active_edges:
                edge_track_dists[ae].append(edge_distances[ae, i])
        else:
            # Activate and track for edge points
            edge_activation[edge_idx] += 1
            edge_track_dists[edge_idx].append(edge_distances[edge_idx, i])

    # Calculate edge lengths
    edge_lengths = np.linalg.norm(solution_edges[:, 1] - solution_edges[:, 0], axis=1)

    # Calculate cost for each edge
    cost = np.zeros(len(solution_edges))
    for i, (edge, activation) in enumerate(zip(solution_edges, edge_activation)):
        if activation == 0:
            # Check neighboring edges
            neighbor_indices = [(i - 1) % len(solution_edges), (i + 1) % len(solution_edges)]
            neighbor_activation = edge_activation[neighbor_indices]

            if np.all(neighbor_activation == 0):
                # Apply edge penalty if neither neighboring edge is active
                cost[i] = edge_penalty * edge_lengths[i]
            else:
                # Find closest points from neighboring edges
                neighbor_points = np.concatenate([edge_track_dists[ni] for ni in neighbor_indices if edge_track_dists[ni]])
                if len(neighbor_points) > 0:
                    closest_dist = np.min(neighbor_points)
                    cost[i] = closest_dist * edge_lengths[i]
                else:
                    cost[i] = edge_penalty * edge_lengths[i]
        else:
            # Calculate cost for active edges
            weight = 1 / (activation / len(data_points) + 0.01)
            cost[i] = weight * np.sum(edge_track_dists[i]) * edge_lengths[i]

    return np.sum(cost),


def get_neighboring_edges_from_vert(vert_idx, num_verts):
    return [(vert_idx - 1) % num_verts, vert_idx]


def point_segment_distance(points, p1, p2):
    """Calculate the distance from points to a line segment."""
    v = p2 - p1
    t = np.maximum(0, np.minimum(1, np.dot(points - p1, v) / np.dot(v, v)))
    projections = p1 + t[:, np.newaxis] * v
    return np.linalg.norm(points - projections, axis=1)




def custom_mutate(individual, indpb, parameter_set, x_range, y_range):
    # Mutate the part from parameter set
    if random.random() < indpb:
        new_id = random.randint(0, len(parameter_set) - 1)
        individual[0] = new_id
        # new_values = random.choice(parameter_set)
        # individual[:4] = new_values  # Assuming the first 4 are from parameter set

    # Mutate x_offset
    if random.random() < indpb:
        individual[1] = random.uniform(*x_range)

    # Mutate y_offset
    if random.random() < indpb:
        individual[2] = random.uniform(*y_range)

    return individual,



def plot_all_generations_hof_and_pareto_front(all_individuals, hof, pareto_front, objective_names, plot_pareto_surface=True):
    """
    Plot all solutions from all generations, highlight the Hall of Fame,
    show the true Pareto front, and optionally plot a Pareto surface using Plotly.

    Parameters:
    all_individuals (list): List of all individuals from all generations
    hof (deap.tools.HallOfFame): Hall of Fame object
    objective_names (list): List of strings with names for each objective
    plot_pareto_surface (bool): If True, plot a semi-transparent Pareto surface

    Returns:
    plotly.graph_objs._figure.Figure: The created Plotly figure
    """
    # Extract fitness values for all solutions
    all_fitness_values = np.array([ind.fitness.values for ind in all_individuals])

    # Extract fitness values for Hall of Fame
    hof_fitness_values = np.array([ind.fitness.values for ind in hof])

    # Extract the true Pareto front
    # pareto_front = tools.sortNondominated(all_individuals, len(all_individuals), first_front_only=True)[0]
    pareto_fitness_values = np.array([ind.fitness.values for ind in pareto_front[0]])

    # Create the scatter plot for all solutions
    fig = go.Figure(data=go.Scatter3d(
        x=all_fitness_values[:, 0],
        y=all_fitness_values[:, 1],
        z=all_fitness_values[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=all_fitness_values[:, 2],  # Color by the third objective
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(title="Fitness (Obj 3)")
        ),
        text=[f"Solution {i}<br>Obj1: {v[0]:.4f}<br>Obj2: {v[1]:.4f}<br>Obj3: {v[2]:.4f}"
              for i, v in enumerate(all_fitness_values)],
        hoverinfo='text',
        name='All Solutions'
    ))

    # Add Hall of Fame solutions
    fig.add_trace(go.Scatter3d(
        x=hof_fitness_values[:, 0],
        y=hof_fitness_values[:, 1],
        z=hof_fitness_values[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color='yellow',
            symbol='diamond',
            line=dict(color='black', width=1)
        ),
        text=[f"HoF Solution {i}<br>Obj1: {v[0]:.4f}<br>Obj2: {v[1]:.4f}<br>Obj3: {v[2]:.4f}"
              for i, v in enumerate(hof_fitness_values)],
        hoverinfo='text',
        name='Hall of Fame'
    ))

    # Add true Pareto front solutions
    fig.add_trace(go.Scatter3d(
        x=pareto_fitness_values[:, 0],
        y=pareto_fitness_values[:, 1],
        z=pareto_fitness_values[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            symbol='circle',
            line=dict(color='black', width=1)
        ),
        text=[f"Pareto Solution {i}<br>Obj1: {v[0]:.4f}<br>Obj2: {v[1]:.4f}<br>Obj3: {v[2]:.4f}"
              for i, v in enumerate(pareto_fitness_values)],
        hoverinfo='text',
        name='True Pareto Front'
    ))

    # Add Pareto surface if requested
    if plot_pareto_surface and len(pareto_fitness_values) > 3:
        # Create a triangulation of the Pareto front points
        tri = Delaunay(pareto_fitness_values[:, :2])

        # Create the mesh for the surface
        xx, yy = np.meshgrid(np.linspace(pareto_fitness_values[:, 0].min(), pareto_fitness_values[:, 0].max(), 100),
                             np.linspace(pareto_fitness_values[:, 1].min(), pareto_fitness_values[:, 1].max(), 100))
        zz = np.zeros(xx.shape)

        # Interpolate z values
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                point = [xx[i, j], yy[i, j]]
                simplex = tri.find_simplex(point)
                if simplex != -1:
                    b = tri.transform[simplex, :2].dot(point - tri.transform[simplex, 2])
                    bary = np.c_[b, 1 - b.sum(axis=-1)]
                    zz[i, j] = np.dot(bary, pareto_fitness_values[tri.simplices[simplex], 2])
                else:
                    zz[i, j] = np.nan

        # Add the surface to the plot
        fig.add_trace(go.Surface(
            x=xx,
            y=yy,
            z=zz,
            colorscale='Reds',
            opacity=0.6,
            name='Pareto Surface'
        ))

    # Update the layout
    fig.update_layout(
        title='3D Visualization of All Solutions, Hall of Fame, and Pareto Front',
        scene=dict(
            xaxis_title=objective_names[0],
            yaxis_title=objective_names[1],
            zaxis_title=objective_names[2],
        ),
        width=1000,
        height=800,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    return fig



def eaMuPlusLambda_history(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None,
                           halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    all_individuals = population[:]  # Store initial population

    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Store all individuals from this generation
        all_individuals.extend(offspring)

        # Update the statistics with the new population
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook, all_individuals



if __name__ == "__main__":
    dummy_individual = [0.02, 0.02, 0.2, 0.2, 0.5, 0.5]
    with open('/Users/fnoic/Library/CloudStorage/OneDrive-TUM/temp/beam_2_dump_2D.txt', 'rb') as f:
        data = np.loadtxt(f)
    fitness = fitness_function_2d_2(dummy_individual, data)
    fitness = efficient_cost_function(dummy_individual, data)
    print(fitness)

    a = 0