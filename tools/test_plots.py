import matplotlib.pyplot as plt
import numpy as np


def plot_test_in(lines, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for line in lines:
        ax.plot([float(line[1]), float(line[4])], [float(line[2]), float(line[5])])
    ax.set_aspect('equal')
    plt.title('test in\n' + path)
    plt.show()

def plot_test_out(skeleton, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for line in skeleton.bones:
        ax.plot(
            [line.left[0], line.right[0]],
            [line.left[1], line.right[1]]
        )
    ax.set_aspect('equal')
    plt.title('test out\n' + path)
    plt.show()

    return None