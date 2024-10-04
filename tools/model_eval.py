import os

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from tqdm import tqdm

from tools.visual import dist_hist, mesh_points_cc, dist_hist_color


def compute_point_cloud_to_mesh_distance(points, mesh):
    # Convert mesh vertices to numpy array
    mesh_points = np.asarray(mesh.vertices)

    # Create KDTree from mesh vertices
    tree = KDTree(mesh_points)

    # Query KDTree for nearest neighbors
    distances, _ = tree.query(points)

    return distances


def model_evaluation(skeleton):
    path_path = "/Users/fnoic/Downloads/exported_ifc.txt"

    with open(path_path, 'r') as file:
        path_obj = file.read()
        path_obj = path_obj[:-4]

    iter = 0
    collected_points = None

    for bone in tqdm(skeleton.bones, desc="Model Evaluation", total=len(skeleton.bones)):

        bone_id = bone.name.split("_")[1]

        for file in os.listdir(path_obj):
            if file.endswith(".obj"):
                obj_id = file[:-4].split("_")[2]
                if bone_id == obj_id:
                    mesh = o3d.io.read_triangle_mesh(f'{path_obj}/{file}')
                    # repair mesh
                    # mesh.remove_duplicated_vertices()
                    # mesh.remove_unreferenced_vertices()
                    # mesh.remove_degenerate_triangles()
                    # mesh.remove_duplicated_triangles()

                    points = np.asarray(bone.points)

                    if iter == 0:
                        collected_points = points
                    else:
                        collected_points = np.concatenate((collected_points, points), axis=0)
                    iter += 1

                    # check if mesh can be processed
                    if not mesh.has_triangles():
                        print(f"Mesh {file} is not intact")
                        continue

                    distances_np = compute_point_cloud_to_mesh_distance(points, mesh)

                    # dist_hist(distances_np, bone_id)
                    dist_hist_color(distances_np, bone_id)
                    mesh_points_cc(points, distances_np, mesh, ortho=True)

    # complete model and complete point cloud
    collected_mesh = o3d.io.read_triangle_mesh(path_obj + '.obj')
    collected_distances = compute_point_cloud_to_mesh_distance(collected_points, collected_mesh)
    dist_hist_color(collected_distances, "complete")
    mesh_points_cc(collected_points, collected_distances, collected_mesh, ortho=True)