import os

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from plotly import graph_objects as go
from scipy.spatial import Delaunay


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


def segment_projection_2D(points, ransac_highlight=False, ransac_data=None):
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
    ax.scatter(0, 0, color='orange', s=10, zorder=10)
    plot_length = 1e3
    origin = [0, 0]
    # plot lines
    ax.plot(
        [-plot_length, plot_length], [0, 0],
        color='red',
        alpha=0.25,
        zorder=5,
        linewidth=4
    )
    ax.plot(
        [0, 0], [-plot_length, plot_length],
        color='green',
        alpha=0.25,
        zorder=5,
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
        opacity=0.25,
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
        opacity=0.5,
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


def dist_hist_color(distances, distances_log, bone_id, angle_desc=None):
    # Create the figure and axis
    fig, ax = plt.subplots()

    # Create the histogram
    n, bins, patches = ax.hist(distances, bins=100) #, edgecolor='black')

    # Create a normalization object for the color mapping
    norm = Normalize(vmin=min(distances_log), vmax=max(distances_log))

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

    if angle_desc is not None:
        ax.set_title(f"Beam {bone_id} - {angle_desc}")

    # Add colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Distance')

    plt.show()


def transformation_tracer(points_source=None, points_target=None, points_inter_1=None, points_inter_2=None,
                          source_angle=None, target_angle=None, source_angle_2=None, target_angle_2=None):
    # create go figure
    fig = go.Figure()
    if points_source is not None:
        # scatter plot source points
        fig.add_trace(go.Scatter3d(
            x=points_source[:, 0], y=points_source[:, 1], z=points_source[:, 2],
            mode='markers',
            marker=dict(size=1, color='blue', opacity=0.8),
            name='Source'
        ))
    if points_target is not None:
        # scatter plot target points
        fig.add_trace(go.Scatter3d(
            x=points_target[:, 0], y=points_target[:, 1], z=points_target[:, 2],
            mode='markers',
            marker=dict(size=1, color='green', opacity=0.8),
            name='Target'
        ))
    if points_inter_1 is not None:
        fig.add_trace(go.Scatter3d(
            x=points_inter_1[:, 0], y=points_inter_1[:, 1], z=points_inter_1[:, 2],
            mode='markers',
            marker=dict(size=1, color='orange', opacity=0.8),
            name='Intermediate 1'
        ))
    if points_inter_2 is not None:
        fig.add_trace(go.Scatter3d(
            x=points_inter_2[:, 0], y=points_inter_2[:, 1], z=points_inter_2[:, 2],
            mode='markers',
            marker=dict(size=1, color='purple', opacity=0.8),
            name='Intermediate 2'
        ))
    if source_angle is not None:
        # add purple line for source angle 0:1
        fig.add_trace(go.Scatter3d(
            x=[source_angle[0][0], source_angle[1][0]], y=[source_angle[0][1], source_angle[1][1]],
            z=[source_angle[0][2], source_angle[1][2]],
            mode='lines',
            line=dict(color='purple', width=10),
            name='Source Angle'
        ))
        # add orange line for source angle 1:2
        fig.add_trace(go.Scatter3d(
            x=[source_angle[1][0], source_angle[2][0]], y=[source_angle[1][1], source_angle[2][1]],
            z=[source_angle[1][2], source_angle[2][2]],
            mode='lines',
            line=dict(color='orange', width=10),
            name='Source Angle'
        ))

    if source_angle_2 is not None:
        # same as source_angle but dotted
        fig.add_trace(go.Scatter3d(
            x=[source_angle_2[0][0], source_angle_2[1][0]], y=[source_angle_2[0][1], source_angle_2[1][1]],
            z=[source_angle_2[0][2], source_angle_2[1][2]],
            mode='lines',
            line=dict(color='purple', width=10, dash='dash'),
            name='Source Angle 2'
        ))
        fig.add_trace(go.Scatter3d(
            x=[source_angle_2[1][0], source_angle_2[2][0]], y=[source_angle_2[1][1], source_angle_2[2][1]],
            z=[source_angle_2[1][2], source_angle_2[2][2]],
            mode='lines',
            line=dict(color='orange', width=10, dash='dash'),
            name='Source Angle 2'
        ))

    if target_angle is not None:
        # add purple line for target angle 0:1
        fig.add_trace(go.Scatter3d(
            x=[target_angle[0][0], target_angle[1][0]], y=[target_angle[0][1], target_angle[1][1]],
            z=[target_angle[0][2], target_angle[1][2]],
            mode='lines',
            line=dict(color='purple', width=10),
            name='Target Angle'
        ))
        # add orange line for target angle 1:2
        fig.add_trace(go.Scatter3d(
            x=[target_angle[1][0], target_angle[2][0]], y=[target_angle[1][1], target_angle[2][1]],
            z=[target_angle[1][2], target_angle[2][2]],
            mode='lines',
            line=dict(color='orange', width=10),
            name='Target Angle'
        ))

    if target_angle_2 is not None:
        # same as target_angle but dotted
        fig.add_trace(go.Scatter3d(
            x=[target_angle_2[0][0], target_angle_2[1][0]], y=[target_angle_2[0][1], target_angle_2[1][1]],
            z=[target_angle_2[0][2], target_angle_2[1][2]],
            mode='lines',
            line=dict(color='purple', width=10, dash='dash'),
            name='Target Angle 2'
        ))
        fig.add_trace(go.Scatter3d(
            x=[target_angle_2[1][0], target_angle_2[2][0]], y=[target_angle_2[1][1], target_angle_2[2][1]],
            z=[target_angle_2[1][2], target_angle_2[2][2]],
            mode='lines',
            line=dict(color='orange', width=10, dash='dash'),
            name='Target Angle 2'
        ))

    if target_angle is None and target_angle_2 is None:
        # add lines for coordinate system
        fig.add_trace(go.Scatter3d(
            x=[0, 1], y=[0, 0], z=[0, 0],
            mode='lines',
            line=dict(color='red', width=10),
            name='X'
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 1], z=[0, 0],
            mode='lines',
            line=dict(color='green', width=10),
            name='Y'
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, 1],
            mode='lines',
            line=dict(color='blue', width=10),
            name='Z'
        ))
    # orthographic projection
    fig.update_layout(scene_camera=dict(projection=dict(type='orthographic')))
    # equal axes
    fig.update_layout(scene=dict(aspectmode='data'))
    # show
    fig.show()


def plot_all_generations_hof_and_pareto_front(all_individuals, hof=None, pareto_front=None, objective_names=None,
                                              plot_pareto_surface=False, ngen=None):
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
    if hof is not None:
        hof_fitness_values = np.array([ind.fitness.values for ind in hof])

    # Extract the true Pareto front
    # pareto_front = tools.sortNondominated(all_individuals, len(all_individuals), first_front_only=True)[0]
    pareto_fitness_values = np.array([ind.fitness.values for ind in pareto_front])

    if ngen is not None:
        fitness_values = np.array([ind.fitness.values for ind in all_individuals])
        pop_size = len(fitness_values) // ngen
        cc_content = np.repeat(np.arange(ngen), pop_size)[:len(fitness_values)]
    else:
        cc_content = all_fitness_values[:, 0]

    # Create the scatter plot for all solutions
    fig = go.Figure(data=go.Scatter3d(
        x=all_fitness_values[:, 0],
        y=all_fitness_values[:, 1],
        z=all_fitness_values[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=cc_content,
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(title="fitness (objective 1)")
        ),
        text=[f"solution {i}<br>obj1: {v[0]:.4f}<br>obj2: {v[1]:.4f}<br>obj3: {v[2]:.4f}"
              for i, v in enumerate(all_fitness_values)],
        hoverinfo='text',
        name='all Solutions'
    ))

    # Add Hall of Fame solutions
    if hof is not None:
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
        z=pareto_fitness_values[:, 2], # cosine instead
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
        margin=dict(r=20, b=10, l=10, t=40),
        # font times new roman
        font=dict(
            family="Times New Roman",
            size=12,
        )
    )

    return fig


def cs_plot(vertices=None, points=None, normals=None, headline=None, save=False, filename=None, iter=None, info_dict=None, weights=None):
    normals = np.asarray(normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, np.finfo(float).eps, norms) * 20
    normals = normals / norms
    points = np.asarray(points)

    # Create figure and adjust size if table exists
    fig = plt.figure(figsize=(4, 6 if info_dict else 6))

    if info_dict:
        # Table subplot
        ax_table = plt.subplot2grid((5, 1), (0, 0), rowspan=1)
        ax_table.axis('off')

        # Create table data
        cells = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)]
                 for k, v in info_dict.items()]

        table = ax_table.table(cellText=cells,
                               colLabels=['Parameter', 'Value'],
                               cellLoc='left',
                               loc='center',
                               bbox=[0.1, 0, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # Main plot
        ax = plt.subplot2grid((5, 1), (1, 0), rowspan=4)
    else:
        ax = fig.add_subplot(111)

    # Original plotting code
    color = 'purple'
    if vertices is not None:
        for i in range(11):
            ax.plot([vertices[i][0], vertices[i + 1][0]],
                    [vertices[i][1], vertices[i + 1][1]], color=color)
        ax.plot([vertices[11][0], vertices[0][0]],
                [vertices[11][1], vertices[0][1]], color=color)

    if points is not None and weights is None:
        try:
            ax.scatter(points[:, 0], points[:, 1], s=0.8, color='grey', marker=',')
        except Exception as e:
            print(e)
    elif points is not None and weights is not None:
        # scatter as above but color with weights / viridis
        try:
            ax.scatter(points[:, 0], points[:, 1], s=0.9, c=weights, cmap='viridis', marker=',')
        except Exception as e:
            print(e)

    if normals is not None and weights is None:
        for i in range(points.shape[0]):
            ax.plot([points[i][0], points[i][0] + normals[i][0] * 0.5],
                    [points[i][1], points[i][1] + normals[i][1] * 0.5],
                    color='grey', linewidth=0.3)
    elif normals is not None and weights is not None:
        # normals as above but color with weights / viridis
        for i in range(points.shape[0]):
            cmap = plt.cm.viridis
            norm = plt.Normalize(min(weights), max(weights))

            ax.plot([points[i][0], points[i][0] + normals[i][0] * 0.5],
                    [points[i][1], points[i][1] + normals[i][1] * 0.5],
                    c=cmap(norm(weights[i])),
                    linewidth=0.5)

    if headline is not None:
        ax.set_title(headline, fontname='Times New Roman')

    ax.set_aspect('equal')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
        item.set_fontname('Times New Roman')

    # fig.set_dpi(300)
    plt.tight_layout()

    if save:
        dir = f'/Users/fnoic/Downloads/tracer/{iter}/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = f'/Users/fnoic/Downloads/tracer/{iter}/{filename}.png'
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
        plt.close()


def rg_plot_active(cloud, active_point_ids, counter_patch):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    # non-segment points
    nonseg_ids = list(set(cloud['id'].to_list()) - set(active_point_ids))
    ax.scatter(cloud.loc[nonseg_ids, 'x'], cloud.loc[nonseg_ids, 'y'], cloud.loc[nonseg_ids, 'z'], c='grey', s=0.1)
    ax.scatter(cloud.loc[active_point_ids, 'x'], cloud.loc[active_point_ids, 'y'], cloud.loc[active_point_ids, 'z'], c='r', s=0.1)
    fig.suptitle(f'Cluster {counter_patch}')
    plt.show()


def rg_plot_what(cloud, active_point_ids, active_plot, point_ids, cluster_sn, cluster_rn,
                 neighbor_patch_sn, color, neighbor_patch_rn, neighbor_patch_csn,
                 deviation_sn, deviation_sn_old, deviation_rn, cluster_confidence,
                 plot_count, config):
    fig = plt.figure(figsize=(20, 20))

    # Subplot 1: Original perspective
    ax1 = fig.add_subplot(221, projection='3d')
    create_scatter(cloud, active_plot, point_ids, cluster_sn, active_point_ids, cluster_rn,
                   neighbor_patch_sn, color, neighbor_patch_rn, neighbor_patch_csn, ax1, 30, 30)

    # Subplot 2: Top-down view
    ax2 = fig.add_subplot(222, projection='3d')
    create_scatter(cloud, active_plot, point_ids, cluster_sn, active_point_ids, cluster_rn,
                   neighbor_patch_sn, color, neighbor_patch_rn, neighbor_patch_csn, ax2, 90, 0)

    # Subplot 3: Side view (YZ plane)
    ax3 = fig.add_subplot(223, projection='3d')
    create_scatter(cloud, active_plot, point_ids, cluster_sn, active_point_ids, cluster_rn,
                   neighbor_patch_sn, color, neighbor_patch_rn, neighbor_patch_csn, ax3, 0, 0)

    # Subplot 4: Front view (XZ plane)
    ax4 = fig.add_subplot(224, projection='3d')
    create_scatter(cloud, active_plot, point_ids, cluster_sn, active_point_ids, cluster_rn,
                   neighbor_patch_sn, color, neighbor_patch_rn, neighbor_patch_csn, ax4, 0, 90)

    fig.suptitle(f'supernormal with context (csn) deviation: {deviation_sn:.2f}°\n'
                 f'supernormal (sn) deviation: {deviation_sn_old:.2f}°\n'
                 f'ransac normal (rn) deviation: {deviation_rn:.2f}°\n'
                 f'cluster confidence: {cluster_confidence:.2f}\n')
    plt.tight_layout()
    plotpath = f'{config.project.basepath_macos}{config.project.project_path}plot/plot_{plot_count}.png'
    plot_count += 1
    plt.savefig(plotpath, dpi=300)
    plt.close()

def create_scatter(cloud, active_plot, point_ids, cluster_sn, active_point_ids, cluster_rn,
                   neighbor_patch_sn, color, neighbor_patch_rn, neighbor_patch_csn, ax, view_elev, view_azim):
    rest_idx = list(set(cloud['id'].to_list()) - set(active_plot) - set(point_ids))
    ax.scatter(cloud.loc[active_plot, 'x'], cloud.loc[active_plot, 'y'], cloud.loc[active_plot, 'z'], c='b', s=0.2)
    ax.scatter(cloud.loc[point_ids, 'x'], cloud.loc[point_ids, 'y'], cloud.loc[point_ids, 'z'], c=color, s=0.2)
    ax.scatter(cloud.loc[rest_idx, 'x'], cloud.loc[rest_idx, 'y'], cloud.loc[rest_idx, 'z'], c='grey', s=0.1, alpha=0.2)

    # Calculate centroid of active points
    centroid = cloud.loc[active_point_ids, ['x', 'y', 'z']].mean().values

    # Create line representing cluster_sn
    end_point = centroid + .5 * cluster_sn
    ax.plot([centroid[0], end_point[0]],
            [centroid[1], end_point[1]],
            [centroid[2], end_point[2]],
            color='b', linewidth=2)

    # Create line representing cluster_rn, dashed
    end_point = centroid + .5 * cluster_rn
    ax.plot([centroid[0], end_point[0]],
            [centroid[1], end_point[1]],
            [centroid[2], end_point[2]],
            color='b', linewidth=2, linestyle='dashed')

    # create line representing neighbor_patch_sn
    end_point = centroid + .5 * neighbor_patch_sn
    ax.plot([centroid[0], end_point[0]],
            [centroid[1], end_point[1]],
            [centroid[2], end_point[2]],
            color=color, linewidth=2)

    # create line representing neighbor_patch_rn, dashed
    end_point = centroid + .5 * neighbor_patch_rn
    ax.plot([centroid[0], end_point[0]],
            [centroid[1], end_point[1]],
            [centroid[2], end_point[2]],
            color=color, linewidth=2, linestyle='dashed')

    # create line for cluster_csn, dotted, purple
    end_point = centroid + .5 * neighbor_patch_csn
    ax.plot([centroid[0], end_point[0]],
            [centroid[1], end_point[1]],
            [centroid[2], end_point[2]],
            color='orange', linewidth=2, linestyle='dotted')

    ax.view_init(elev=view_elev, azim=view_azim)

    return ax