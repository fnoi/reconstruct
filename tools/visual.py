import numpy as np
import matplotlib.pyplot as plt


def vis_segment_planes_3D(segment, point, planes, proj_plane):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(segment.points[:, 0], segment.points[:, 1], segment.points[:, 2], marker='.', s=0.01)
    thresh = 1
    xlim = (np.min(segment.points[:, 0]) - thresh, np.max(segment.points[:, 0]) + thresh)
    ylim = (np.min(segment.points[:, 1]) - thresh, np.max(segment.points[:, 1]) + thresh)
    zlim = (np.min(segment.points[:, 2]) - thresh, np.max(segment.points[:, 2]) + thresh)

    x = np.linspace(xlim[0], xlim[1], 3)
    y = np.linspace(ylim[0], ylim[1], 3)
    z = np.linspace(zlim[0], zlim[1], 3)

    x1, y1 = np.meshgrid(x, y)
    a1, b1, c1, d1 = planes[0]
    z1 = (- a1 * x1 - b1 * y1 - d1) / c1
    ax.plot_surface(x1, y1, z1, alpha=0.3)

    x2, z2 = np.meshgrid(x, z)
    a2, b2, c2, d2 = planes[1]
    y2 = (- a2 * x2 - c2 * z2 - d2) / b2
    ax.plot_surface(x2, y2, z2, alpha=0.3)

    y2, z2 = np.meshgrid(y, z)
    a2, b2, c2, d2 = proj_plane
    x2 = (- b2 * y2 - c2 * z2 - d2) / a2
    ax.plot_surface(x2, y2, z2, alpha=0.3)

    ax.scatter(segment.left[0], segment.left[1], segment.left[2], marker='o', s=10)
    ax.scatter(point[0], point[1], point[2], marker='o', s=10)

    ax.set_xlim = xlim
    ax.set_ylim = ylim
    ax.set_zlim = zlim

    fig.show()


def vis_segment_projection_A(proj_pts_2, linepts_0_0, linepts_0_1, linepts_1_0, linepts_1_1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(proj_pts_2[:, 0], proj_pts_2[:, 1], proj_pts_2[:, 2])
    # ax.scatter(points_on_line[:, 0], points_on_line[:, 1], points_on_line[:, 2], color='red')
    ax.plot(
        [linepts_0_0[0], linepts_0_1[0]],
        [linepts_0_0[1], linepts_0_1[1]],
        [linepts_0_0[2], linepts_0_1[2]],
        color='red')
    ax.plot(
        [linepts_1_0[0], linepts_1_1[0]],
        [linepts_1_0[1], linepts_1_1[1]],
        [linepts_1_0[2], linepts_1_1[2]],
        color='purple')
    ax.set_aspect('equal')
    fig.show()

def vis_segment_projection_B(rotated_pts, rotated_linepts):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(rotated_pts[:, 0], rotated_pts[:, 1], s=0.05)
    ax.plot(
        [rotated_linepts[0, 0], rotated_linepts[1, 0]],
        [rotated_linepts[0, 1], rotated_linepts[1, 1]],
        color='red')
    ax.plot(
        [rotated_linepts[2, 0], rotated_linepts[3, 0]],
        [rotated_linepts[2, 1], rotated_linepts[3, 1]],
        color='purple')
    ax.set_aspect('equal')
    fig.show()