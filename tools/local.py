import copy
import random

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

from tools.utils import plot_patch


def supernormal_svd(normals):
    U, S, Vt = np.linalg.svd(normals)
    return Vt[-1, :]


def supernormal_confidence(supernormal, normals):
    """
    calculate confidence value of supernormal,
    confidence = md_sn / md_n
    md_sn: mean deviation angle between supernormal and normals (-90)
    md_n:  mean deviation angle between normals
    big sn, small n -> big confidence
    big sn, big n   -> small confidence
    small sn, big n -> very small confidence
    small sn, small n -> big confidence
    (is the idea, lets check)
    """
    supernormal /= np.linalg.norm(supernormal)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    angles = np.arccos(np.dot(supernormal, normals.T))
    angles = np.abs(np.rad2deg(angles))
    angles = np.abs(angles - 90)
    angles = np.clip(angles, 0, 90)
    normalized_angles = angles / 90
    c = np.average(1 - normalized_angles)
    c *= len(normals)

    return c


def angular_deviation(vector, reference):
    norm_vector = np.linalg.norm(vector)
    norm_reference = np.linalg.norm(reference)
    if norm_vector == 0 or norm_reference == 0:
        raise ValueError("Input vectors must be non-zero.")

    vector_normalized = vector / norm_vector
    reference_normalized = reference / norm_reference

    # compute dot product and clamp it to the valid range for arccos
    dot_product = np.dot(vector_normalized, reference_normalized)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # compute the angle in radians and then convert to degrees
    angle = np.arccos(dot_product)
    return angle * 180 / np.pi


def neighborhood_calculations(cloud=None, seed_id=None, config=None, plot_ind=None, plot_flag=False):
    neighbor_ids = neighborhood_search(cloud, seed_id, config)

    if plot_flag and seed_id == plot_ind:
        neighborhood_plot(cloud, seed_id, neighbor_ids, config)

    neighbor_normals = cloud.iloc[neighbor_ids][['nx', 'ny', 'nz']].values
    seed_supernormal = supernormal_svd(neighbor_normals)
    seed_supernormal /= np.linalg.norm(seed_supernormal)

    cloud.loc[seed_id, 'snx'] = seed_supernormal[0]
    cloud.loc[seed_id, 'sny'] = seed_supernormal[1]
    cloud.loc[seed_id, 'snz'] = seed_supernormal[2]
    cloud.loc[seed_id, 'confidence'] = supernormal_confidence(seed_supernormal, neighbor_normals)

    return cloud


def calculate_supernormals_rev(cloud=None, cloud_o3d=None, config=None):
    plot_ind = random.randint(0, len(cloud))
    plot_ind = 762
    print(f'plot ind is {plot_ind}')
    plot_flag = True

    point_ids = np.arange(len(cloud))

    for seed_id in tqdm(point_ids, desc="computing supernormals", total=len(point_ids)):
        if config.local_features.neighbor_shape in ['cube', 'sphere', 'ellipsoid']:  # unoriented neighborhoods
            cloud = neighborhood_calculations(cloud=cloud, seed_id=seed_id, config=config,
                                              plot_ind=plot_ind, plot_flag=plot_flag)
            # no second step of computation needed
        elif config.local_features.neighbor_shape in ['oriented_ellipsoid', 'oriented_cylinder', 'oriented_cuboid']:
            real_config = copy.copy(config)  # save config
            config.local_features.neighbor_shape = "sphere"  # override for precomputation
            cloud = neighborhood_calculations(cloud=cloud, seed_id=seed_id, config=config,
                                              plot_ind=plot_ind, plot_flag=plot_flag)
            config = real_config  # reset config
            # oriented neighborhoods require supernormals as input
            cloud = neighborhood_calculations(cloud=cloud, seed_id=seed_id, config=config,
                                              plot_ind=plot_ind, plot_flag=plot_flag)

        else:
            raise ValueError(f'neighborhood shape "{config.local_features.neighbor_shape}" not implemented')

    return cloud


def ransac_patches(cloud, config):
    print(f'ransac patching')
    cloud['rnx'] = 0.0
    cloud['rny'] = 0.0
    cloud['rnz'] = 0.0
    cloud['ransac_patch'] = 0
    mask_remaining = np.ones(len(cloud), dtype=bool)
    progress = tqdm()
    blanks = config.clustering.ransac_blanks

    label_id = 0  # label 0 indicates unclustered
    while True:
        o3d_cloud_current = o3d.geometry.PointCloud()
        o3d_cloud_current.points = o3d.utility.Vector3dVector(cloud.loc[mask_remaining, ['x', 'y', 'z']].values)

        # ransac plane fitting
        ransac_plane, ransac_inliers = o3d_cloud_current.segment_plane(
            distance_threshold=config.clustering.ransac_dist_thresh,
            ransac_n=config.clustering.ransac_n,
            num_iterations=config.clustering.ransac_iterations
        )
        ransac_normal = ransac_plane[0:3]
        inliers_global_idx = np.where(mask_remaining)[0][ransac_inliers]

        # dbscan clustering of inliers
        dbscan_clustering = DBSCAN(
            eps=config.clustering.dbscan_eps_dist,
            min_samples=config.clustering.dbscan_min_count
        ).fit(cloud.loc[inliers_global_idx, ['x', 'y', 'z']].values)
        active_idx = np.where(mask_remaining)[0]

        if len(np.unique(dbscan_clustering.labels_)) == 1:
            blanks -= 1
            if blanks == 0:
                print(f'over, {label_id} patches found')
                break

        for cluster in np.unique(dbscan_clustering.labels_[dbscan_clustering.labels_ != -1]):
            # these points form a valid cluster
            # find the index of the points in the original cloud
            label_id += 1
            cluster_idx = np.where(dbscan_clustering.labels_ == cluster)[0]
            external_cluster_idx = active_idx[ransac_inliers][cluster_idx]
            cloud.loc[external_cluster_idx, 'ransac_patch'] = label_id
            cloud.loc[external_cluster_idx, 'rnx'] = ransac_normal[0]
            cloud.loc[external_cluster_idx, 'rny'] = ransac_normal[1]
            cloud.loc[external_cluster_idx, 'rnz'] = ransac_normal[2]
            mask_remaining[external_cluster_idx] = False

        progress.update()

    debug_plot = True
    if debug_plot:
        # plot histogram for ransac_patch
        plt.hist(cloud['ransac_patch'], bins=label_id)
        plt.show()

        # create plot with two subplots
        ax_lims_x = (np.min(cloud['x']), np.max(cloud['x']))
        ax_lims_y = (np.min(cloud['y']), np.max(cloud['y']))
        ax_lims_z = (np.min(cloud['z']), np.max(cloud['z']))

        plt.figure()
        mask_done = np.invert(mask_remaining)
        cloud_clustered = cloud.loc[mask_done]
        ax1 = plt.subplot(121, projection='3d')
        ax1.scatter(cloud_clustered['x'], cloud_clustered['y'], cloud_clustered['z'],
                    c=plt.cm.tab10(np.mod(cloud_clustered['ransac_patch'], 10)),
                    s=.5)
        ax1.set_xlim(ax_lims_x)
        ax1.set_ylim(ax_lims_y)
        ax1.set_zlim(ax_lims_z)

        cloud_unclustered = cloud.loc[mask_remaining]
        ax2 = plt.subplot(122, projection='3d')
        ax2.scatter(cloud_unclustered['x'], cloud_unclustered['y'], cloud_unclustered['z'], c='grey', s=.5)
        ax1.set_xlim(ax_lims_x)
        ax1.set_ylim(ax_lims_y)
        ax1.set_zlim(ax_lims_z)

        # set tight
        plt.tight_layout()
        plt.show()

    return cloud


def patch_growing(cloud, config):
    patches_in = np.unique(cloud['ransac_patch'])
    patches_out = []

    while True:
        seed = int(cloud.idxmax()['confidence'])


def region_growing_rev(cloud, config):
    mask_remaining = np.ones(len(cloud), dtype=bool)
    progress = tqdm()
    label_id = 0
    clustered_patches = []
    cloud['instance_pred'] = None

    while True:  # top level loop
        seed = int(cloud.idxmax()['confidence'])
        cluster_active = [seed]
        cluster_passive = []
        candidates_failed = []
        add_candidates = []
        patch_additions = []
        angle_checked = []
        patch_failed = []

        while True:  # mid level loop
            # growth check, clean up temps???
            growth_plot(cloud, seed, cluster_active, cluster_passive, add_candidates, patch_additions, angle_checked)
            # add patch for all (added) neighbors
            for new_id in cluster_active:
                patch_id = cloud.loc[new_id, 'ransac_patch']
                if patch_id not in clustered_patches and patch_id != 0 and patch_id not in patch_failed:
                    # check patch angle deviation to seed  # TODO: alternative: mean cluster sn, needs testing
                    patch_sn = np.mean(cloud[cloud['ransac_patch'] == patch_id][['snx', 'sny', 'snz']].values, axis=0)
                    sn_deviation = angular_deviation(
                        np.abs(patch_sn),
                        np.abs(cloud.loc[seed, ['snx', 'sny', 'snz']].values)
                    )
                    if sn_deviation <= config.region_growing.supernormal_patch_angle_deviation:
                        clustered_patches.append(patch_id)
                        patch_additions.extend(cloud[cloud['ransac_patch'] == patch_id].index)
                        cluster_active.extend(cloud[cloud['ransac_patch'] == patch_id].index)
                    else:
                        patch_failed.append(patch_id)
            # check plot after patches have been added
            growth_plot(cloud, seed, cluster_active, cluster_passive, add_candidates, patch_additions, angle_checked)
            # identify patch neighborhood
            add_candidates = []
            for new_id in cluster_active:
                add_candidates.extend(neighborhood_search(cloud, new_id, config))
            add_candidates = np.unique(add_candidates)
            add_candidates = [x for x in add_candidates
                              if x not in cluster_active
                              and x not in cluster_passive
                              and x not in angle_checked]
            if len(add_candidates) <= config.region_growing.neighbors_found_min:
                label_id += 1
                cloud.loc[cluster_active, 'instance_pred'] = label_id
                break
            # check plot for neighborhood logic
            growth_plot(cloud, seed, cluster_active, cluster_passive, add_candidates, patch_additions, angle_checked)
            # check if the neighbors are cool
            for candidate in add_candidates:
                sn_deviation = angular_deviation(cloud.loc[candidate, ['snx', 'sny', 'snz']].values,
                                                 cloud.loc[seed, ['snx', 'sny', 'snz']].values)
                if sn_deviation <= config.region_growing.supernormal_angle_deviation:
                    cluster_active.append(candidate)
                else:
                    angle_checked.append(candidate)

            add_candidates = []
            patch_additions = []

            growth_plot(cloud, seed, cluster_active, cluster_passive, add_candidates, patch_additions, angle_checked)

            a = 0

        # encode id to cloud
        # check remaining ptx
        # break criterion


def neighborhood_search(cloud, seed_id, config):
    seed_data = cloud.iloc[seed_id]
    match config.local_features.neighbor_shape:
        case "sphere":
            cloud_tree = KDTree(cloud[['x', 'y', 'z']].values)
            neighbor_ids = cloud_tree.query_ball_point([seed_data['x'], seed_data['y'], seed_data['z']],
                                                       r=config.local_features.supernormal_radius)
        case "cylinder":
            neighbor_ids = neighbors_oriented_cylinder(cloud, seed_id, config)
        case "cuboid":
            neighbor_ids = neighbors_oriented_cuboid(cloud, seed_id, config)
        case "ellipsoid":
            neighbor_ids = neighbors_ellipsoid(cloud, seed_id, config)
        case "cube":
            neighbor_ids = neighbors_aabb_cube(cloud, seed_id, config)
        case "oriented_ellipsoid":
            neighbor_ids = neighbors_oriented_ellipsoid(cloud, seed_id, config)
        case "oriented_octahedron":
            neighbor_ids = neighbors_oriented_octahedron(cloud, seed_id, config)  # TODO: add this fct
        case _:
            raise ValueError(f'neighborhood shape "{config.local_features.neighbor_shape}" not implemented')

    return neighbor_ids


def growth_plot(cloud, seed_id, cluster_active, cluster_passive, add_candidates, patch_additions, angle_checked):
    # create mask for cloud to exclude any id in clustered_inactive, add_candidates, patch_additions
    mask = np.ones(len(cloud), dtype=bool)

    mask[cluster_active] = False
    mask[add_candidates] = False
    mask[patch_additions] = False
    mask[cluster_passive] = False
    mask[angle_checked] = False
    mask[seed_id] = False

    # remove patch additions from cluster_active
    cluster_active = [x for x in cluster_active if x not in patch_additions]

    fig = plt.figure()

    subplots = [221, 222, 223, 224]
    perspectives = [(0, 0), (90, 90), (45, 225), (45, 315)]
    _ax = [fig.add_subplot(subplot, projection='3d') for subplot in subplots]

    for ax_ in _ax:
        ax_.scatter(cloud.loc[mask, 'x'],
                    cloud.loc[mask, 'y'],
                    cloud.loc[mask, 'z'],
                    s=0.2, c='grey')
        ax_.scatter(cloud.loc[cluster_passive, 'x'],
                    cloud.loc[cluster_passive, 'y'],
                    cloud.loc[cluster_passive, 'z'],
                    s=0.6, c='grey')
        ax_.scatter(cloud.loc[cluster_active, 'x'],
                    cloud.loc[cluster_active, 'y'],
                    cloud.loc[cluster_active, 'z'],
                    s=0.6, c='blue')
        ax_.scatter(cloud.loc[patch_additions, 'x'],
                    cloud.loc[patch_additions, 'y'],
                    cloud.loc[patch_additions, 'z'],
                    s=0.6, c='green')
        ax_.scatter(cloud.loc[add_candidates, 'x'],
                    cloud.loc[add_candidates, 'y'],
                    cloud.loc[add_candidates, 'z'],
                    s=0.6, c='yellow')
        ax_.scatter(cloud.loc[angle_checked, 'x'],
                    cloud.loc[angle_checked, 'y'],
                    cloud.loc[angle_checked, 'z'],
                    s=0.6, c='red')
        ax_.scatter(cloud.loc[seed_id, 'x'],
                    cloud.loc[seed_id, 'y'],
                    cloud.loc[seed_id, 'z'],
                    s=10, c='orange')
        ax_.view_init(elev=perspectives[_ax.index(ax_)][0], azim=perspectives[_ax.index(ax_)][1])
        ax_.set_aspect('equal')
        ax_.set_xticks([])
        ax_.set_yticks([])
        ax_.set_zticks([])

    # make a legend for the colors and force the dots larger
    fig.legend(['unclustered', 'cluster_passive', 'cluster_active', 'patch_additions', 'add_candidates', 'angle_checked', 'seed'],
               loc='center left', bbox_to_anchor=(0.4, 0.5))
    # make bigger
    fig.tight_layout()
    fig.set_size_inches(7, 7)
    plt.show()


def neighborhood_plot(cloud, seed_id=None, neighbors=None, config=None, cage_override=None, confidence=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cloud['x'], cloud['y'], cloud['z'], s=0.3, c='grey')

    if neighbors is not None:
        ax.scatter(cloud.loc[neighbors, 'x'], cloud.loc[neighbors, 'y'], cloud.loc[neighbors, 'z'], s=1, c='r')

    if confidence:
        ax.scatter(cloud['x'], cloud['y'], cloud['z'], s=0.3, c=cloud['confidence'], cmap='viridis')

    cage_color = 'b'
    cage_width = 0.5
    if cage_override is not None:
        config = None

    if config is not None:
        match config.local_features.neighbor_shape:
            case "cube":
                # cornerpoints from seed and supernormal_cube_dist as edge length of cube
                center = cloud.loc[seed_id, ['x', 'y', 'z']].values
                edge_length = config.local_features.supernormal_cube_dist
                p0 = center + np.array([-edge_length, -edge_length, -edge_length])
                p1 = center + np.array([-edge_length, -edge_length, edge_length])
                p2 = center + np.array([-edge_length, edge_length, -edge_length])
                p3 = center + np.array([-edge_length, edge_length, edge_length])
                p4 = center + np.array([edge_length, -edge_length, -edge_length])
                p5 = center + np.array([edge_length, -edge_length, edge_length])
                p6 = center + np.array([edge_length, edge_length, -edge_length])
                p7 = center + np.array([edge_length, edge_length, edge_length])
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], c=cage_color, linewidth=cage_width)
                ax.plot([p1[0], p3[0]], [p1[1], p3[1]], [p1[2], p3[2]], c=cage_color, linewidth=cage_width)
                ax.plot([p3[0], p2[0]], [p3[1], p2[1]], [p3[2], p2[2]], c=cage_color, linewidth=cage_width)
                ax.plot([p2[0], p0[0]], [p2[1], p0[1]], [p2[2], p0[2]], c=cage_color, linewidth=cage_width)
                ax.plot([p4[0], p5[0]], [p4[1], p5[1]], [p4[2], p5[2]], c=cage_color, linewidth=cage_width)
                ax.plot([p5[0], p7[0]], [p5[1], p7[1]], [p5[2], p7[2]], c=cage_color, linewidth=cage_width)
                ax.plot([p7[0], p6[0]], [p7[1], p6[1]], [p7[2], p6[2]], c=cage_color, linewidth=cage_width)
                ax.plot([p6[0], p4[0]], [p6[1], p4[1]], [p6[2], p4[2]], c=cage_color, linewidth=cage_width)
                ax.plot([p0[0], p4[0]], [p0[1], p4[1]], [p0[2], p4[2]], c=cage_color, linewidth=cage_width)
                ax.plot([p1[0], p5[0]], [p1[1], p5[1]], [p1[2], p5[2]], c=cage_color, linewidth=cage_width)
                ax.plot([p2[0], p6[0]], [p2[1], p6[1]], [p2[2], p6[2]], c=cage_color, linewidth=cage_width)
                ax.plot([p3[0], p7[0]], [p3[1], p7[1]], [p3[2], p7[2]], c=cage_color, linewidth=cage_width)
            case "sphere":
                # plot sphere around seed point
                n_segments = 14
                center = cloud.loc[seed_id, ['x', 'y', 'z']].values
                radius = config.local_features.supernormal_radius
                u = np.linspace(0, 2 * np.pi, n_segments)
                v = np.linspace(0, np.pi, n_segments)
                x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
                y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
                z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
                # ax.plot_surface(x, y, z, color='b', alpha=0.1)
                # wireframe
                for i in range(n_segments):
                    ax.plot(x[i, :], y[i, :], z[i, :], c=cage_color, linewidth=cage_width)
                    ax.plot(x[:, i], y[:, i], z[:, i], c=cage_color, linewidth=cage_width)

            case "ellipsoid":
                # plot ellipsoid around seed point
                n_segments = 14
                center = cloud.loc[seed_id, ['x', 'y', 'z']].values
                a = config.local_features.supernormal_ellipsoid_a  # Semi-major axis (along x)
                b = config.local_features.supernormal_ellipsoid_bc  # Semi-minor axes (along y and z)
                u = np.linspace(0, 2 * np.pi, n_segments)
                v = np.linspace(0, np.pi, n_segments)
                x = a * np.outer(np.cos(u), np.sin(v)) + center[0]
                y = b * np.outer(np.sin(u), np.sin(v)) + center[1]
                z = b * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
                # ax.plot_surface(x, y, z, color='b', alpha=0.1)
                # wireframe
                for i in range(n_segments):
                    ax.plot(x[i, :], y[i, :], z[i, :], c=cage_color, linewidth=cage_width)
                    ax.plot(x[:, i], y[:, i], z[:, i], c=cage_color, linewidth=cage_width)

            case "oriented_ellipsoid":
                n_segments = 14
                center = cloud.loc[seed_id, ['x', 'y', 'z']].values
                a = config.local_features.supernormal_ellipsoid_a  # Semi-major axis (along x)
                b = config.local_features.supernormal_ellipsoid_bc  # Semi-minor axes (along y and z)
                u = np.linspace(0, 2 * np.pi, n_segments)
                v = np.linspace(0, np.pi, n_segments)
                x = a * np.outer(np.cos(u), np.sin(v))
                y = b * np.outer(np.sin(u), np.sin(v))
                z = b * np.outer(np.ones(np.size(u)), np.cos(v))

                direction = cloud.loc[seed_id, ['snx', 'sny', 'snz']].values

                rot_mat = find_orthonormal_basis(direction)

                ellipsoid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
                rotated_points = rot_mat @ ellipsoid_points

                x_rotated = rotated_points[0, :].reshape(x.shape) + center[0]
                y_rotated = rotated_points[1, :].reshape(y.shape) + center[1]
                z_rotated = rotated_points[2, :].reshape(z.shape) + center[2]

                for i in range(n_segments):
                    ax.plot(x_rotated[i, :], y_rotated[i, :], z_rotated[i, :], c=cage_color, linewidth=cage_width)
                    ax.plot(x_rotated[:, i], y_rotated[:, i], z_rotated[:, i], c=cage_color, linewidth=cage_width)

            case _:
                print(
                    f'\nno cage plot implemented for the neighborhood shape of {config.local_features.neighbor_shape}')

    ax.set_aspect('equal')

    if seed_id is not None:
        ax.scatter(cloud.loc[seed_id, 'x'], cloud.loc[seed_id, 'y'], cloud.loc[seed_id, 'z'], s=10, c='orange')

    plt.show()


def find_orthonormal_basis(direction):
    """
    find orthonormal basis from direction vector
    """
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    if np.allclose(direction, [0, 0, 1]) or np.allclose(direction, [0, 1, 0]):
        other = np.array([1, 0, 1])
    else:
        other = np.array([0, 0, 1])
    y_axis = np.cross(direction, other)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(direction, y_axis)

    return np.column_stack((direction, y_axis, z_axis))


def neighbors_oriented_ellipsoid(cloud, seed_id, config):
    """
    find neighbors of seed_id in ellipsoid shape
    """
    seed_data = cloud.iloc[seed_id]
    seed_coords = seed_data[['x', 'y', 'z']].values
    cloud_coords = cloud[['x', 'y', 'z']].values
    a = config.local_features.supernormal_ellipsoid_a
    b = config.local_features.supernormal_ellipsoid_bc  # Semi-minor axis along y and z

    orientation = seed_data[['snx', 'sny', 'snz']].values
    rot_mat = find_orthonormal_basis(orientation)

    rotated_cloud = (cloud_coords - seed_coords) @ rot_mat
    rotated_seed_coords = np.array([0, 0, 0])

    # calculate squared distances normalized by the squared semi-axis lengths
    x_dist_norm = (rotated_cloud[:, 0] / a) ** 2
    y_dist_norm = (rotated_cloud[:, 1] / b) ** 2
    z_dist_norm = (rotated_cloud[:, 2] / b) ** 2

    neighbor_ids = np.where(x_dist_norm + y_dist_norm + z_dist_norm <= 1)[0]

    neighbor_ids = neighbor_ids[neighbor_ids != seed_id]

    return neighbor_ids


def neighbors_ellipsoid(cloud, seed_id, config):
    """
    find neighbors of seed_id within an ellipsoidal shape.
    """
    seed_data = cloud.iloc[seed_id]
    coordinates_seed = seed_data[['x', 'y', 'z']].values
    coordinates_cloud = cloud[['x', 'y', 'z']].values

    a = config.local_features.supernormal_ellipsoid_a
    b = config.local_features.supernormal_ellipsoid_bc  # Semi-minor axis along y and z

    # Calculate squared distances normalized by the squared semi-axis lengths
    x_dist_norm = ((coordinates_cloud[:, 0] - coordinates_seed[0]) / a) ** 2
    y_dist_norm = ((coordinates_cloud[:, 1] - coordinates_seed[1]) / b) ** 2
    z_dist_norm = ((coordinates_cloud[:, 2] - coordinates_seed[2]) / b) ** 2

    # Sum the normalized distances and find indices where the sum is less than or equal to 1
    neighbor_ids = np.where(x_dist_norm + y_dist_norm + z_dist_norm <= 1)[0]

    # Remove seed_id from neighbor_ids to exclude the point itself
    neighbor_ids = neighbor_ids[neighbor_ids != seed_id]

    return neighbor_ids


def neighbors_oriented_cylinder(cloud, seed_id, config):
    """
    find neighbors of seed_id in cylinder shape
    """
    seed_data = cloud.iloc[seed_id]

    return idx


def neighbors_oriented_cuboid(cloud, seed_id, config):
    """
    find neighbors of seed_id in cuboid shape
    """
    seed_data = cloud.iloc[seed_id]

    return idx


def neighbors_aabb_cube(cloud, seed_id, config):
    """
    find neighbors of seed_id in axis aligned bounding box shape
    """
    seed_data = cloud.iloc[seed_id]
    coordinates_seed = seed_data[['x', 'y', 'z']].values
    coordinates_cloud = cloud[['x', 'y', 'z']].values
    dist = config.local_features.supernormal_cube_dist

    # find neighbors in x direction
    neighbor_ids = np.where(np.abs(coordinates_cloud[:, 0] - coordinates_seed[0]) < dist)[0]
    # find neighbors in y direction
    neighbor_ids = np.intersect1d(neighbor_ids,
                                  np.where(np.abs(coordinates_cloud[:, 1] - coordinates_seed[1]) < dist)[0])
    # find neighbors in z direction
    neighbor_ids = np.intersect1d(neighbor_ids,
                                  np.where(np.abs(coordinates_cloud[:, 2] - coordinates_seed[2]) < dist)[0])

    # remove seed_id from neighbor_ids
    neighbor_ids = neighbor_ids[neighbor_ids != seed_id]

    return neighbor_ids
