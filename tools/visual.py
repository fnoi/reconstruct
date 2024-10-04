import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


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

    ax.scatter(segment.line_raw_left[0], segment.line_raw_left[1], segment.line_raw_left[2], marker='o', s=10)
    ax.scatter(point[0], point[1], point[2], marker='o', s=10)

    ax.set_xlim = xlim
    ax.set_ylim = ylim
    ax.set_zlim = zlim

    fig.show()


def segment_projection_3D(points, lines):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[0], points[1], points[2], s=10, c='orange')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, alpha=0.4)
    # ax.scatter(points_on_line[:, 0], points_on_line[:, 1], points_on_line[:, 2], color='red')
    # ax.plot(
    #     lines[:, 0], lines[:, 1], lines[:, 2],
    #     color='green'
    # )
    colors = ['red', 'purple']
    for color, line in zip(colors, lines):
        ax.plot(
            line[:, 0], line[:, 1], line[:, 2],
            color=color
        )
        # )
    # ax.plot(
    #     lines[0:1, 0], lines[0:1, 1], lines[0:1, 2],
    #     color='red')
    # ax.plot(lines[1:2, 0], lines[1:2, 1], lines[1:2, 2],
    #     color='purple')
    ax.set_aspect('equal')
    # put perspective to x axis
    ax.view_init(elev=0, azim=0)
    fig.show()


def segment_projection_2D(points, lines, extra_point=None, ransac_highlight=False, ransac_data=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mask_scatter = np.ones(points.shape[0], dtype=bool)
    if ransac_highlight:
        mask_scatter[ransac_data[0]] = False
        mask_scatter[ransac_data[1]] = False
    ax.scatter(points[mask_scatter, 0], points[mask_scatter, 1], s=0.05, color='grey', zorder=7)
    if ransac_highlight:
        ax.scatter(points[ransac_data[0], 0], points[ransac_data[0], 1], s=0.05, color='red', zorder=9)
        ax.scatter(points[ransac_data[1], 0], points[ransac_data[1], 1], s=0.05, color='purple', zorder=8)
    if extra_point is not None:
        ax.scatter(extra_point[0], extra_point[1], color='orange', s=10, zorder=10)
    line_0 = np.array([extra_point - lines[0] * 1e3, extra_point + lines[0] * 1e3])
    line_1 = np.array([extra_point - lines[1] * 1e3, extra_point + lines[1] * 1e3])
    # plot lines
    ax.plot(
        line_0[:, 0], line_0[:, 1],
        color='red',
        alpha=0.25,
        zorder=5,
        linewidth=4
    )
    ax.plot(
        line_1[:, 0], line_1[:, 1],
        color='purple',
        alpha=0.25,
        zorder=6,
        linewidth=4
    )
    # limit axis to points + 10%
    rel_ext = 0.1
    x_ext = np.abs(np.max(points[:, 0]) - np.min(points[:, 0]))
    y_ext = np.abs(np.max(points[:, 1]) - np.min(points[:, 1]))
    xmid = np.min(points[:, 0]) + (x_ext / 2)
    ymid = np.min(points[:, 1]) + (y_ext / 2)
    if x_ext > y_ext:
        xlim = xmid - (x_ext/2) - (rel_ext * x_ext), xmid + (x_ext/2) + (rel_ext * x_ext)
        ylim = ymid - (x_ext/2) - (rel_ext * x_ext), ymid + (x_ext/2) + (rel_ext * x_ext)
    else:
        ylim = ymid - (y_ext/2) - (rel_ext * y_ext), ymid + (y_ext/2) + (rel_ext * y_ext)
        xlim = xmid - (y_ext/2) - (rel_ext * y_ext), xmid + (y_ext/2) + (rel_ext * y_ext)
    ax.set_xlim(xmin=xlim[0], xmax=xlim[1])
    ax.set_ylim(ymin=ylim[0], ymax=ylim[1])

    fig.show()


def mesh_to_plotly(mesh):
    # Extract triangle indices and vertices
    triangle_ids = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # Initialize lists for Plotly's Mesh3d
    i, j, k = [], [], []

    x = list(vertices[:, 0])
    y = list(vertices[:, 1])
    z = list(vertices[:, 2])

    # Initialize set to keep track of unique edges
    edges = set()
    edges_vertices = []

    x_e, y_e, z_e = [], [], []

    for triangle in triangle_ids:
        # define vertex coordinates
        i.append(triangle[0])
        j.append(triangle[1])
        k.append(triangle[2])

        if len(set(triangle)) != 3:
            a = 0
            continue

        # Add unique edges
        for edge in [(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])]:
            if edge[0] == edge[1]:
                continue
            else:
                edge = tuple(sorted(edge))  # Ensure consistent ordering
                if edge not in edges:
                    edges.add(edge)
                    edges_vertices.append((vertices[edge[0]], vertices[edge[1]]))

                    x_e.extend([vertices[edge[0]][0], vertices[edge[1]][0], None])
                    y_e.extend([vertices[edge[0]][1], vertices[edge[1]][1], None])
                    z_e.extend([vertices[edge[0]][2], vertices[edge[1]][2], None])

    return i, j, k, x, y, z, x_e, y_e, z_e


def mesh_points_cc(points, distances, mesh, ortho=False):
    i, j, k, x, y, z, x_edge, y_edge, z_edge = mesh_to_plotly(mesh)
    # mesh is o3d mesh!
    mesh_np = np.asarray(mesh.vertices)
    # plot mesh semi transparent with edges in plotly
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(    # TODO: make robust for separate objs and add edges option
        x=x,
        y=y,
        z=z,
        opacity=0.5,
        color='grey',
        i=i,
        j=j,
        k=k)
    )
    # plot edges
    fig.add_trace(go.Scatter3d(
        x=x_edge,
        y=y_edge,
        z=z_edge,
        mode='lines',
        line=dict(color='black', width=1)
    ))

    # scatter points in 3D, color by distance
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=distances,
            colorscale='Viridis'
        )
    ))
    # equal aspect ratio
    fig.update_layout(scene=dict(aspectmode='data'))
    # perspective
    if ortho:
        fig.update_layout(scene_camera=dict(projection=dict(type='orthographic')))

    # no axes, no grid
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))

    fig.show()


def dist_hist(distances, bone_id):
    # plot histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(distances, bins=100)
    ax.set_title(f"Beam {bone_id}")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")
    plt.show()


def dist_hist_color(distances, bone_id):
    # Create the figure and axis
    fig, ax = plt.subplots()

    # Create the histogram
    n, bins, patches = ax.hist(distances, bins=100) #, edgecolor='black')

    # Create a normalization object for the color mapping
    norm = Normalize(vmin=min(distances), vmax=max(distances))

    # Create a ScalarMappable object for the colorbar
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])

    # Color each bar according to its central value
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for count, patch, center in zip(n, patches, bin_centers):
        color = sm.cmap(norm(center))
        patch.set_facecolor(color)

    # Set labels and title
    ax.set_title(f"Beam {bone_id}")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")

    # Add colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Distance')

    plt.show()