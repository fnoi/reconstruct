import copy
import pickle
import random
import time

import numpy as np
from deap import creator, base, tools, algorithms
from numpy.lib.npyio import savez

from tools.fitting_1 import params2verts, verts2edges

from sklearn.metrics.pairwise import cosine_similarity

from tools.visual import plot_all_generations_hof_and_pareto_front, cs_plot


def solve_w_nsga(points, normals, config, all_points, all_normals, cs_data, cs_dataframe,
                 filter_weights=None, filter_map=None):
    x_range, y_range = setup_lims_placement(points)

    points_array = copy.deepcopy(points)
    # points_array = np.asarray(points_array)
    points = [row.tolist() for row in points]
    normals = [row.tolist() for row in normals]

    ###########
    # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))                 # log_distance, active_edge_length_relative, activation_distance
    # creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0, 1.0)) # minimize log distance, maximize relative active edge length, minimize edge activation distance, maximize cosine sim
    # creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    # Register the function to select a random row from the parameter set
    toolbox.register("attr_row", random.randint, 0, len(cs_data) - 1)
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

    toolbox.register("evaluate", cost_combined, data_points=points, data_normals=normals, data_frame=cs_dataframe)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutate, indpb=0.2, parameter_set=cs_data, x_range=x_range, y_range=y_range)
    # toolbox.register("select", tools.selTournament, tournsize=3)

    # toolbox.register("select", tools.selNSGA2)
    ref_points = tools.uniform_reference_points(nobj=4, p=12) ####!
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    population = toolbox.population(n=config.cs_fit.n_pop)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # alternative with deap eaMuPlusLambda
    # final_pop, log = algorithms.eaMuPlusLambda(
    #     population, toolbox,
    #     mu=config.cs_fit.n_mu, lambda_=config.cs_fit.n_lambda, cxpb=0.5, mutpb=0.2,
    #     ngen=config.cs_fit.n_gen, stats=stats, halloffame=hof, verbose=True
    # )

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

    pareto_front = tools.sortNondominated(hof, len(hof), first_front_only=True)
    # pareto_front = tools.sortNondominated(all_individuals, len(all_individuals), first_front_only=True)
    # pareto_front = tools.sortNondominated(final_pop, len(final_pop), first_front_only=True)

    pareto_unique = []
    for pareto_individual in pareto_front[0]:
        if pareto_individual not in pareto_unique:
            pareto_unique.append(pareto_individual)

    min_fitness_individual_abs = min(pareto_unique, key=lambda x: x.fitness.values[0])

    pareto_plot = True
    if pareto_plot:
        fig = plot_all_generations_hof_and_pareto_front(all_individuals,
                                                        hof,
                                                        pareto_unique,
                                                        ["Log Distance", "Active Edge Length", "Cosine Similarity"],
                                                        plot_pareto_surface=False
                                                        )
        fig.show()


    params = min_fitness_individual_abs[1:] + cs_data[min_fitness_individual_abs[0]]
    verts = params2verts(params, from_cog=False)
    edges = verts2edges(verts)
    dists = np.array([point_segment_distance(points_array, edge[0], edge[1]) for edge in edges])
    inliers = np.sum(np.min(dists, axis=0) < config.cs_fit.inlier_thresh)
    rel_inliers = inliers / len(points)
    cs_plot(vertices=verts,
            points=points,
            normals=normals,
            headline=f'best solution: {inliers}\ninliers: log only, relative: {rel_inliers:.2f}')

    # for each solution in pareto_unique, plot to file with headline indicating all fitness values
    store_iter = time.strftime("%Y%m%d-%H%M%S")
    for pareto_individual in pareto_unique:
        params = pareto_individual[1:] + cs_data[pareto_individual[0]]
        verts = params2verts(params, from_cog=False)
        edges = verts2edges(verts)
        dists = np.array([point_segment_distance(points_array, edge[0], edge[1]) for edge in edges])
        inliers = np.sum(np.min(dists, axis=0) < config.cs_fit.inlier_thresh)
        rel_inliers = inliers / len(points)
        cs_plot(vertices=verts,
                points=all_points,
                normals=all_normals,
                headline=f'best solution: {inliers}\ninliers: log only, relative: {rel_inliers:.2f}'
                         f'\nfitness values: {pareto_individual.fitness.values}',
                save=True,
                filename=f'pareto_{pareto_individual[0]}',
                iter=store_iter)

    # calculate inliers for all in pareto and identify solution with max inliers
    inliers_all = []
    for pareto_individual in pareto_unique:
        params = pareto_individual[1:] + cs_data[pareto_individual[0]]
        verts = params2verts(params, from_cog=False)
        edges = verts2edges(verts)
        dists = np.array([point_segment_distance(points_array, edge[0], edge[1]) for edge in edges])
        inliers = np.sum(np.min(dists, axis=0) < config.cs_fit.inlier_thresh)
        inliers_all.append(inliers)
    max_inliers = max(inliers_all)
    max_inliers_idx = inliers_all.index(max_inliers)

    params = pareto_unique[max_inliers_idx][1:] + cs_data[pareto_unique[max_inliers_idx][0]]
    verts = params2verts(params, from_cog=False)
    edges = verts2edges(verts)
    inliers = max_inliers / len(points)
    cs_plot(vertices=verts,
            points=all_points,
            normals=all_normals,
            headline=f'best solution: {max_inliers_idx}\ninliers: inliers cutoff, relative: {inliers:.2f}')



    # delete Individual and FitnessMin
    del creator.FitnessMulti
    del creator.Individual

    # should return h_beam_params, h_beam_verts and final cost
    h_beam_params = params
    h_beam_verts = verts
    h_beam_cost = min_fitness_individual_abs.fitness.values

    # cstype is the name of the profile
    cstype = cs_dataframe.iloc[min_fitness_individual_abs[0]]['name']

    # h_beam_cost no longer returned

    return h_beam_params, h_beam_verts, cstype



def point_segment_distance(points, edge_start, edge_end):
    edge_vec = edge_end - edge_start
    edge_len_sq = np.sum(edge_vec ** 2)

    if edge_len_sq == 0:
        return np.linalg.norm(points - edge_start, axis=1)

    t = np.sum((points - edge_start) * edge_vec, axis=1) / edge_len_sq
    t = np.clip(t, 0, 1)

    projection = edge_start + t[:, None] * edge_vec

    return np.linalg.norm(points - projection, axis=1)


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


def cost_combined(solution_params, data_points, data_normals, data_frame, weights=None, weights_map=None):
    data_points = np.array(data_points)
    data_normals = np.array(data_normals)

    params = data_frame.iloc[solution_params[0]]
    params = params[['tw', 'tf', 'bf', 'd']].values.tolist()
    solution_params = [solution_params[1], solution_params[2]] + params
    solution_verts = params2verts(solution_params, from_cog=False)
    solution_edges = verts2edges(solution_verts)

    n_up = [0, 1]
    n_down = [0, -1]
    n_left = [-1, 0]
    n_right = [1, 0]

    polygon_edge_normals = np.array(
        [n_left, n_up, n_left, n_down, n_left, n_up, n_right, n_down, n_right, n_up, n_right, n_down]
    )

    # Pre-compute all cosine similarities
    all_similarities = np.array([
        cosine_similarity(normal.reshape(1, -1), data_normals)[0]
        for normal in polygon_edge_normals
    ])
    if weights is not None:
        all_similarities = all_similarities * weights

    edge_lengths = np.linalg.norm(solution_edges[:, 1] - solution_edges[:, 0], axis=1)
    edge_length_total = np.sum(edge_lengths)

    edge_distances = np.array([point_segment_distance(data_points, edge[0], edge[1])
                                 for edge in solution_edges])
    if weights is not None:
        edge_distances = edge_distances * weights_map

    best_edge_per_point = np.argmin(edge_distances, axis=0)
    min_distances_per_point = np.min(edge_distances, axis=0)
    log_distance = np.sum(np.log(min_distances_per_point)) / len(data_points)
    # lin_distance = np.sum(min_distances_per_point) / len(data_points)

    edge_activity = np.zeros(len(solution_edges))
    edge_activation_dist = np.zeros(len(solution_edges))

    for edge in range(len(solution_edges)):
        if edge in best_edge_per_point:
            edge_activity[edge] = 1
    edge_activity_log = copy.deepcopy(edge_activity)
    edge_quality = 0

    cosine_similarity_n_n = np.zeros_like(min_distances_per_point)

    # normal_cosine_similarity = 0
    for edge in range(len(solution_edges)):


        if edge_activity_log[edge] == 1:
            related_points = np.where(best_edge_per_point == edge)[0]
            cosine_similarity_n_n = all_similarities[edge][related_points]

            normal_cosine_similarity = np.sum(all_similarities[edge][related_points])

            edge_quality += normal_cosine_similarity

            neighbor_low = edge - 1 if edge - 1 >= 0 else len(solution_edges) - 1
            neighbor_high = edge + 1 if edge + 1 < len(solution_edges) else 0
            active_low = edge_activity_log[neighbor_low] == 0
            active_high = edge_activity_log[neighbor_high] == 0

            if active_low or active_high:
                edge_activity[edge] = 1 # double check bc big change
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

    # normal_cosine_similarity = normal_cosine_similarity / len(solution_edges)
    if weights is not None:
        cosine_similarity_n_n = cosine_similarity_n_n * weights
    mean_cosines_similarity = np.mean(cosine_similarity_n_n)

    active_edge_length_relative = np.sum(edge_activity * edge_lengths) / edge_length_total
    activation_distance = np.sum(edge_activation_dist)

    # mean_edge_cosine_quality = np.mean(edge_quality)
    mean_edge_cosine_quality = edge_quality

    return log_distance, active_edge_length_relative, activation_distance, mean_cosines_similarity #mean_edge_cosine_quality #normal_cosine_similarity



if __name__ == "__main__":
    dummy_individual = [0.02, 0.02, 0.2, 0.2, 0.5, 0.5]
    with open('/Users/fnoic/Library/CloudStorage/OneDrive-TUM/temp/beam_2_dump_2D.txt', 'rb') as f:
        data = np.loadtxt(f)
    fitness = cost_combined(dummy_individual, data, data)
    print(fitness)

    a = 0


