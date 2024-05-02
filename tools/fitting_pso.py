import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyswarm import pso

def costFunctionGirder(sol, points):
    # Assume simplified calculation just for demonstration
    x0, y0, tf, tw, lf, lw = sol
    vertices = np.array([
        [x0, y0],
        [x0 + lf, y0],
        [x0 + lf, y0 + tf],
        [x0, y0 + tf],
        [x0, y0]  # Closing the loop for visualization
    ])
    RMSE = np.sqrt(np.mean((np.mean(vertices, axis=0) - points)**2))
    return RMSE

# Parameters and PSO setup
lower_bounds = [1, 1, 0.1, 0.1, 1, 1]
upper_bounds = [5, 5, 1, 1, 3, 3]
points = np.array([[2, 2], [3, 3], [4, 4]])

positions = []
def objective_with_logging(x, *args):
    positions.append(x.copy())
    return costFunctionGirder(x, *args)

xopt, fopt = pso(objective_with_logging, lower_bounds, upper_bounds, args=(points,), swarmsize=30, maxiter=50)

# Animation setup
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
line, = ax.plot([], [], 'r-o')  # Ensure markers show
scat = ax.scatter(points[:, 0], points[:, 1], color='blue')

def init():
    line.set_data([], [])
    return line, scat,

def update(frame):
    sol = positions[frame]
    x0, y0, tf, tw, lf, lw = sol
    vertices = np.array([
        [x0, y0],
        [x0 + lf, y0],
        [x0 + lf, y0 + tf],
        [x0, y0 + tf],
        [x0, y0]  # Closing the loop for visualization
    ])
    line.set_data(vertices[:, 0], vertices[:, 1])
    return line, scat,

ani = FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=True, repeat=False)
ani.save('girder_animation.gif', writer='imagemagick', fps=5)
