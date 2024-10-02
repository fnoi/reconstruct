import os

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from tqdm import tqdm


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

    for bone in tqdm(skeleton.bones, desc="Model Evaluation", total=len(skeleton.bones)):
        bone_id = bone.name.split("_")[1]

        for file in os.listdir(path_obj):
            if file.endswith(".obj"):
                obj_id = file[:-4].split("_")[2]
                if bone_id == obj_id:
                    mesh = o3d.io.read_triangle_mesh(f'{path_obj}/{file}')
                    points = np.asarray(bone.points)

                    # check if mesh is intact
                    if not mesh.has_triangles():
                        print(f"Mesh {file} is not intact")
                        continue

                    distances_np = compute_point_cloud_to_mesh_distance(points, mesh)

                    # plot histogram
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.hist(distances_np, bins=100)
                    ax.set_title(f"Bone {bone_id} - Mesh {obj_id}")
                    ax.set_xlabel("Distance")
                    ax.set_ylabel("Frequency")
                    plt.show()