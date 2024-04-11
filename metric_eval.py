import json
import pickle

import numpy as np
import pandas as pd
import open3d as o3d
from matplotlib import pyplot as plt

from omegaconf import OmegaConf

from tools.utils import calculate_miou_with_labels, calculate_view_direction

def plot_debug_fct(cloud_frame, gt_id, pr_id, gt_orientation, pr_orientation):
    gt_instance = cloud_frame[cloud_frame['gj'] == gt_id]
    ptx = gt_instance[['x', 'y', 'z']].to_numpy()
    centroid = np.mean(ptx, axis=0)
    # caluclate max extents of instance in axis direction
    max_extents = np.max(np.max(ptx, axis=0) - np.min(ptx, axis=0))/2
    gt_direction = [
        [centroid[0] - gt_orientation[0] * max_extents, centroid[0] + gt_orientation[0] * max_extents],
        [centroid[1] - gt_orientation[1] * max_extents, centroid[1] + gt_orientation[1] * max_extents],
        [centroid[2] - gt_orientation[2] * max_extents, centroid[2] + gt_orientation[2] * max_extents]
    ]
    pr_direction = [
        [centroid[0] - pr_orientation[0] * max_extents, centroid[0] + pr_orientation[0] * max_extents],
        [centroid[1] - pr_orientation[1] * max_extents, centroid[1] + pr_orientation[1] * max_extents],
        [centroid[2] - pr_orientation[2] * max_extents, centroid[2] + pr_orientation[2] * max_extents]
    ]
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ptx[:, 0], ptx[:, 1], ptx[:, 2], s=0.3, c='grey')
    ax.scatter(centroid[0], centroid[1], centroid[2], s=10, c='g')
    ax.plot(gt_direction[0], gt_direction[1], gt_direction[2], c='r')
    ax.plot(pr_direction[0], pr_direction[1], pr_direction[2], c='b', linestyle='-.')
    ax.set_aspect('equal')
    plt.legend(['instance points', 'instance centroid', 'GT orientation', 'Predicted orientation'])
    plt.title(f'GT instance {gt_id} and predicted instance {pr_id}')
    plt.show()
    a = 0



if __name__ == '__main__':
    # load orientation ground truth
    orientation_gt = OmegaConf.load('instance_orientation.yaml')
    # load orientation prediction
    with open('./data/in_test/dirs.pkl', 'rb') as f:
        orientation_pr = pickle.load(f)
    # load fully annotated point cloud
    with open('./data/in_test/test_junction_segmentation_results.txt', 'r') as f:
        full_cloud = np.loadtxt(f, delimiter=' ')

    cloud_frame = pd.DataFrame(full_cloud)
    cloud_frame.columns = ['x', 'y', 'z', 'c', 'pp', 'pi', 'gj']
    # c: confidence, pp: predicted plane cluster, pi: predicted instance, gj: ground truth instance
    # where pi=3 replace with 0
    cloud_frame['pi'] = cloud_frame['pi'].replace(3, 0)  # TODO: fix in prediction pipeline, min points threshold

    # instance matching and iou calculation
    miou, id_map = calculate_miou_with_labels(cloud_frame.to_numpy())
    print(f'Mean IoU: {miou}')
    angle_values = []
    for id_pair in id_map:
        pr_id = id_pair[0]
        gt_id = id_pair[1]

        # get the orientation of the ground truth instance
        rpy = orientation_gt[str(int(gt_id))]
        rpy = [rpy.X1, rpy.Y2, rpy.Z3]
        gt_orientation = calculate_view_direction(rpy[0], rpy[1], rpy[2])
        # gt_orientationa = [0.0070,0.99998,0.0004]
        # print(gt_orientation - gt_orientationa)
        pr_orientation = orientation_pr[str(pr_id)]
        # ensure directions are normalized and point in the same direction
        gt_orientation = gt_orientation / np.linalg.norm(gt_orientation)
        pr_orientation = pr_orientation / np.linalg.norm(pr_orientation)

        # calculate the smallest angle between the two orientations
        angle = np.arccos(np.dot(gt_orientation, pr_orientation))
        angle_values.append(angle)

        # plot instance points and orientations
        plot_debug = True
        if plot_debug:
            plot_debug_fct(cloud_frame, gt_id, pr_id, gt_orientation, pr_orientation)

        print(f'Angle between GT instance {gt_id} and predicted instance {pr_id}: {angle}')

        # debug_me = True
        if angle > 10:
            with open('./data/in_test/gt_.obj', 'w') as file:
                file.write("v 0.0 0.0 0.0\n")  # Origin
                file.write(f"v {gt_orientation[0]} {gt_orientation[1]} {gt_orientation[2]}\n")  # New gt point
                file.write("l 1 2\n")  # Line from origin to gt point
            with open('./data/in_test/pr_.obj', 'w') as file:
                file.write("v 0.0 0.0 0.0\n")
                file.write(f"v {pr_orientation[0]} {pr_orientation[1]} {pr_orientation[2]}\n")
                file.write("l 1 2\n")
            print('GT and predicted orientations written to file')

    print(f'mean angle: {np.mean(angle_values)}')