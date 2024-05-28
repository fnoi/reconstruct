import numpy as np
from matplotlib import pyplot as plt

def innerdist2edge3(vertices, points):
    num_edges = vertices.shape[0]
    num_points = points.shape[0]
    edge_pts = [[] for _ in range(num_edges)]
    penalty = 100
    activity_dist = np.zeros(num_edges)
    edge_dist = np.zeros(num_edges)

    for i, point in enumerate(points):
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


def plotResultsGirder(BestSol, model):
    fontSize = 12

    # Plot the points
    plt.scatter(model['points'][:, 0], model['points'][:, 1], c='r',
                s=100)  # marker size set to 20 (s=100 for size equivalency)
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X', fontsize=fontSize)
    plt.ylabel('Y', fontsize=fontSize)
    plt.gca().set_aspect('equal', adjustable='box')

    # Extract the solution values
    x0 = BestSol['Position'][0]
    y0 = BestSol['Position'][1]
    tf = BestSol['Position'][2]
    tw = BestSol['Position'][3]
    lf = BestSol['Position'][4]
    lw = BestSol['Position'][5]

    # Calculate vertices
    vertices = np.array([
        [x0, y0],
        [x0 + lf, y0],
        [x0 + lf, y0 + tf],
        [x0 + lf - (lf - tw) / 2, y0 + tf],
        [x0 + lf - (lf - tw) / 2, y0 + tf + lw],
        [x0 + (lf - tw) / 2, y0 + tf + lw],
        [x0 + (lf - tw) / 2, y0 + 2 * tf + lw],
        [x0, y0 + 2 * tf + lw],
        [x0, y0 + tf + lw],
        [x0 + (lf - tw) / 2, y0 + tf + lw],
        [x0 + (lf - tw) / 2, y0 + tf],
        [x0, y0 + tf],
        [x0, y0]
    ])

    # Plot the vertices representing the girder structure
    plt.plot(vertices[:, 0], vertices[:, 1], 'k-o', markerfacecolor='b', linewidth=1.5)
    plt.show()

    # Capturing the plot as an image frame is not directly needed in matplotlib
    # If you need to save or process the figure, you can do so by saving to a file or similar methods
    # plt.savefig("output.png")  # Example: saving to a file


import numpy as np

# Placeholder for auxiliary functions (to be defined)
def Crossover2(p1, p2, VarRange):
    # Implement crossover logic here
    return p1, p2

def Mutate3(p, VarRange):
    # Implement mutation logic here
    return p

def SortPopulation(pop):
    # Sort pop based on the Cost
    return sorted(pop, key=lambda x: x['Cost'])

def fittingUsingPSOGA(params, points, CostFunction):
    VarSize = [1, len(params['lowerBound'])]
    VarMin = np.array(params['lowerBound'])
    VarMax = np.array(params['upperBound'])
    VelMax = (VarMax - VarMin) / 10
    VelMin = -VelMax

    # PSO-GA Parameters
    MaxIt = params['MaxIt']
    nPop = params['nPop']

    # Constriction Coefficients
    phi1, phi2 = 2.05, 2.05
    phi = phi1 + phi2
    chi = 2 / (phi - 2 + np.sqrt(phi**2 - 4 * phi))
    w, c1, c2 = chi, phi1 * chi, phi2 * chi

    # Crossover and Mutation
    pCrossover = 0.9
    nCrossover = int(round(pCrossover * nPop / 2) * 2)
    pMutation = 0.3
    nMutation = int(round(pMutation * nPop))

    # Initialize Population
    pop = [{'Position': np.random.uniform(VarMin, VarMax, VarSize),
            'Velocity': np.zeros(VarSize),
            'Cost': None,
            'Best': {'Position': None, 'Cost': np.inf}} for _ in range(nPop)]

    # Initialize Global Best
    BestSol = {'Cost': np.inf}

    # Evaluate Initial Population
    for particle in pop:
        particle['Cost'] = CostFunction(particle['Position'], points)
        particle['Best']['Position'] = particle['Position']
        particle['Best']['Cost'] = particle['Cost']
        if particle['Best']['Cost'] < BestSol['Cost']:
            BestSol = particle['Best']

    # Main Loop
    for it in range(MaxIt):
        # GA and PSO operations would be performed here
        # Placeholder for GA and PSO logic

        # Store the best cost value
        BestCost = [BestSol['Cost'] for _ in range(MaxIt)]

        # Display iteration info
        print(f"Iteration {it + 1}: Best Cost = {BestCost[it]}")

    return BestSol, BestCost

# Example usage of the function
params = {
    'MaxIt': 100,
    'nPop': 50,
    'lowerBound': [-10, -10, -10],
    'upperBound': [10, 10, 10]
}

# Define a dummy CostFunction
def costFunctionGirder(sol, model):
    points = model['points']
    numPoints = model['numPoints']

    # Extract solution parameters
    x0, y0, tf, tw, lf, lw = sol

    # Calculate vertices based on the solution
    v1 = np.array([x0, y0])
    v2 = np.array([x0 + lf, y0])
    v3 = np.array([x0 + lf, y0 + tf])
    v4 = np.array([x0 + lf - (lf-tw)/2, y0 + tf])
    v5 = np.array([x0 + lf - (lf-tw)/2, y0 + tf + lw])
    v6 = np.array([x0 + (lf-tw)/2, y0 + tf + lw])
    v7 = np.array([x0 + (lf-tw)/2, y0 + 2*tf + lw])
    v8 = np.array([x0, y0 + 2*tf + lw])
    v9 = np.array([x0, y0 + tf + lw])
    v10 = np.array([x0 + (lf-tw)/2, y0 + tf + lw])
    v11 = np.array([x0 + (lf-tw)/2, y0 + tf])
    v12 = np.array([x0, y0 + tf])

    vertices = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12])

    # Call a function to calculate the RMSE between the girder vertices and points
    RMSE = innerdist2edge3(vertices, points)
    return RMSE


