import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde
from sklearn.metrics import average_precision_score, precision_recall_curve

from tools.IO import load_angles


def find_pairs(pred, gt, method):

    if method == 'greedy_gt':
        return find_pairs_greedy_gt(pred, gt)
    elif method == 'greedy_pred':
        return find_pairs_greedy_pred(pred, gt)
    elif method == 'hungarian':
        return find_pairs_hungarian(pred, gt)
    else:
        raise ValueError(f"Invalid method: {method}")


def find_pairs_greedy_gt(pred, gt):
    """
    find the best matching between predicted and ground truth instances using a greedy algorithm
    """
    unique_gt, gt_counts = np.unique(gt, return_counts=True)
    unique_pred, pred_counts = np.unique(pred, return_counts=True)

    # sort uniques by count
    unique_gt = unique_gt[np.argsort(gt_counts)]
    unique_pred = unique_pred[np.argsort(pred_counts)]

    label_pairs = []
    for gt_label in unique_gt:
        if gt_label == 0:
            continue
        best_pred_label = None
        best_iou = 0
        for pred_label in unique_pred:
            intersection = np.sum((pred == pred_label) & (gt == gt_label))
            union = np.sum((pred == pred_label) | (gt == gt_label))
            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_pred_label = pred_label
        if best_pred_label is not None:
            label_pairs.append((best_pred_label, gt_label))
            unique_pred = unique_pred[unique_pred != best_pred_label]

    return label_pairs


def find_pairs_greedy_pred(pred, gt):
    """
    find the best matching between predicted and ground truth instances using a greedy algorithm
    """
    unique_gt, gt_counts = np.unique(gt, return_counts=True)
    unique_pred, pred_counts = np.unique(pred, return_counts=True)

    # sort uniques by count
    unique_gt = unique_gt[np.argsort(gt_counts)]
    unique_pred = unique_pred[np.argsort(pred_counts)]

    label_pairs = []
    for pred_label in unique_pred:
        if pred_label == 0:
            continue
        best_gt_label = None
        best_iou = 0
        for gt_label in unique_gt:
            if gt_label == 0:
                continue
            intersection = np.sum((pred == pred_label) & (gt == gt_label))
            union = np.sum((pred == pred_label) | (gt == gt_label))
            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_gt_label = gt_label
        if best_gt_label is not None:
            label_pairs.append((pred_label, best_gt_label))
            unique_gt = unique_gt[unique_gt != best_gt_label]

    return label_pairs


def find_pairs_hungarian(pred, gt):
    """
    Find the best matching between predicted and ground truth instances using the Hungarian algorithm,
    excluding the unlabeled class (label 0).
    """
    # Filter out the zero labels and get unique labels with their counts
    unique_gt, gt_counts = np.unique(gt[gt != 0], return_counts=True)
    unique_pred, pred_counts = np.unique(pred[pred != 0], return_counts=True)

    # Pad labels to equalize the dimensions of the cost matrix if necessary
    num_gt_pad = max(0, len(unique_pred) - len(unique_gt))
    num_pred_pad = max(0, len(unique_gt) - len(unique_pred))

    pred_labels = np.pad(unique_pred, (0, num_pred_pad), constant_values=0)
    gt_labels = np.pad(unique_gt, (0, num_gt_pad), constant_values=0)
    pred_counts = np.concatenate([pred_counts, np.zeros(num_pred_pad)])
    gt_counts = np.concatenate([gt_counts, np.zeros(num_gt_pad)])

    # Create cost matrix with 1 - IOU as the cost
    cost_matrix = np.zeros((len(pred_labels), len(gt_labels)))
    for i, pred_label in enumerate(pred_labels):
        for j, gt_label in enumerate(gt_labels):
            if pred_label != 0 and gt_label != 0:
                intersection = np.sum((pred == pred_label) & (gt == gt_label))
                union = np.sum((pred == pred_label) | (gt == gt_label))
                iou = intersection / union if union > 0 else 0
                cost_matrix[i, j] = 1 - iou

    # Hungarian algorithm to find the minimum cost assignment (1 - IOU)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Return label pairs excluding those involving the padded zeros if any
    label_pairs = [(pred_labels[i], gt_labels[j]) for i, j in zip(row_ind, col_ind) if
                   pred_labels[i] != 0 and gt_labels[j] != 0]

    return label_pairs


def calculate_precision_recall_iou(pred, gt, id_map, thresholds=None):
    pred = -pred
    for _pred, _gt in id_map:
        pred[pred == -_pred] = _gt
    # pred[pred == 0] = -1
    # gt[gt == 0] = -1

    pr_raw = {}
    for label in np.unique(gt):
        if label == 0:
            continue
        tp = np.sum((pred == label) & (gt == label))
        fp = np.sum((pred == label) & (gt != label))
        tn = np.sum((pred != label) & (gt != label))
        fn = np.sum((pred != label) & (gt == label))

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0

        # count label in gt
        n_gt = np.sum(gt == label)
        n_pred = np.sum(pred == label)

        pr_raw[label] = (precision, recall, iou, n_gt, n_pred)

    if thresholds is None:
        thresholds = [0.0, 0.5]

    gt_total_valid = np.sum(gt != 0)
    pr_thresh = {}
    for threshold in thresholds:
        precision, recall, iou, gt_weighted_iou, gt_weighted_precision = [], [], [], [], []

        for class_label in pr_raw:

            if pr_raw[class_label][2] >= threshold:
                precision.append(pr_raw[class_label][0])
                recall.append(pr_raw[class_label][1])
                iou.append(pr_raw[class_label][2])
                gt_weighted_precision.append(pr_raw[class_label][0] * pr_raw[class_label][3] / gt_total_valid)
                gt_weighted_iou.append(pr_raw[class_label][2] * pr_raw[class_label][3] / gt_total_valid)

            else:
                precision.append(0)
                recall.append(0)
                iou.append(0)
                gt_weighted_precision.append(0)
                gt_weighted_iou.append(0)

        pr_thresh[threshold] = {}
        pr_thresh[threshold]['precision'] = precision
        pr_thresh[threshold]['recall'] = recall
        pr_thresh[threshold]['gt_weighted_precision'] = gt_weighted_precision
        pr_thresh[threshold]['mean_precision'] = np.mean(precision)
        pr_thresh[threshold]['mean_precision_weighted'] = np.sum(gt_weighted_precision)
        pr_thresh[threshold]['mean_iou'] = np.mean(iou)
        pr_thresh[threshold]['mean_iou_weighted'] = np.sum(gt_weighted_iou)

        report_flag = True
        if report_flag:
            print(f"mAP   unweighted  @{threshold}: {pr_thresh[threshold]['mean_precision']:.4f}")
            print(f"mAP   weighted    @{threshold}: {pr_thresh[threshold]['mean_precision_weighted']:.4f}")
            print(f"mIoU  unweighted  @{threshold}: {pr_thresh[threshold]['mean_iou']:.4f}")
            print(f"mIoU  weighted    @{threshold}: {pr_thresh[threshold]['mean_iou_weighted']:.4f} <--")

    if report_flag:
        return pr_thresh
    else:
        # set desired iou threshold here (has to be included in thresholds)
        return pr_thresh[0.0]['mean_iou_weighted'], pr_thresh[0.0]['mean_iou']


def huber_loss(residual, delta):
    if abs(residual) <= delta:
        return 0.5 * residual ** 2
    else:
        return delta * (abs(residual) - 0.5 * delta)


def calculate_metrics(df_cloud, config):
    inst_pred = df_cloud['instance_pr'].to_numpy()
    inst_gt = df_cloud['instance_gt'].to_numpy()

    print('hungarian matching')
    id_map = find_pairs(pred=inst_pred, gt=inst_gt, method='hungarian')
    print(f'mapped pairs {len(id_map)}, {id_map}')
    metrics = calculate_precision_recall_iou(inst_pred, inst_gt, id_map)
    print(metrics)

    greedy_compare = False
    if not greedy_compare:
        return metrics
    else:
        print('greedy_gt matching')
        id_map = find_pairs(pred=inst_pred, gt=inst_gt, method='greedy_gt')
        print(f'mapped pairs {len(id_map)}, {id_map}')
        metrics = calculate_precision_recall_iou(inst_pred, inst_gt, id_map)
        print(metrics)

        print('greedy_pred matching')
        id_map = find_pairs(pred=inst_pred, gt=inst_gt, method='greedy_pred')
        print(f'mapped pairs {len(id_map)}, {id_map}')
        metrics = calculate_precision_recall_iou(inst_pred, inst_gt, id_map)
        print(metrics)

        raise ValueError('undefined return for greedy comparison')


def supernormal_evaluation(cloud, config):
    orientation_gt = load_angles(config.project.orientation_gt_path)
    # orientation_gt = load_angles('instance_orientation.yaml')
    # iterate over rows in cloud
    cloud['supernormal_dev_gt'] = None
    for idx, row in cloud.iterrows():
        # get instance id
        instance_id = row['instance_gt']
        # get orientation from yaml
        gt_orientation = orientation_gt[instance_id]
        # get supernormal
        sn = [row['snx'], row['sny'], row['snz']]
        # calculate angle
        angle = np.rad2deg(np.arccos(np.dot(gt_orientation, sn)))
        cloud.at[idx, 'supernormal_dev_gt_raw'] = angle
        if angle > 90:
            angle = 180 - angle
        # append to list
        cloud.at[idx, 'supernormal_dev_gt'] = angle

    # Set font to Times New Roman for all text
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.sans-serif'] = ['Times New Roman']
    rcParams['font.size'] = 12

    # plot cloud
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Create the scatter plot
    scatter = ax.scatter(cloud['x'], cloud['y'], cloud['z'], s=0.3, c=cloud['supernormal_dev_gt'], cmap='viridis')

    # Get the limits of the data
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Calculate the range of each dimension
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    # Find the greatest range for normalization
    max_range = max(x_range, y_range, z_range)

    # Calculate the midpoints
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    # Set the new limits
    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    plt.show()
    plt.savefig('supernormal_gt_deviation.pdf', bbox_inches='tight', dpi=300)

    # plot cloud with confidence colors
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Create the scatter plot
    scatter = ax.scatter(cloud['x'], cloud['y'], cloud['z'], s=0.3, c=cloud['confidence'], cmap='viridis')

    # Get the limits of the data
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Calculate the range of each dimension
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    # Find the greatest range for normalization
    max_range = max(x_range, y_range, z_range)

    # Calculate the midpoints
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    # Set the new limits
    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

    # Set tick labels to Times New Roman
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        label.set_fontname('Times New Roman')

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    plt.show()


    x = cloud['confidence']
    y = cloud['supernormal_dev_gt']
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots(figsize=(7, 4))

    # Scatter plot with color representing density
    scatter = ax.scatter(x, y, c=z, cmap='viridis', s=1, alpha=0.5)

    # Set y-axis to logarithmic scale
    ax.set_yscale('log')

    # Label axes with LaTeX math notation
    ax.set_xlabel(r'Confidence $c_s$')
    ax.set_ylabel(r'Supernormal Deviation $\theta_\Delta$')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Density', rotation=270, labelpad=15)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    plt.show()
    fig.savefig('supernormal_deviation.pdf', bbox_inches='tight', dpi=300)

    # calculate mean and median deviation
    mean_dev = np.mean(cloud['supernormal_dev_gt'])
    median_dev = np.median(cloud['supernormal_dev_gt'])
    print(f'mean deviation: {mean_dev:.2f} degrees, median deviation: {median_dev:.2f} degrees')


def normal_evaluation(cloud, config):
    orientation_gt = load_angles(config.project.orientation_gt_path)
    # iterate over rows in cloud
    cloud['normal_dev_gt'] = None
    for idx, row in cloud.iterrows():
        # get instance id
        instance_id = row['instance_gt']
        # get orientation from yaml
        gt_orientation = orientation_gt[instance_id]
        # get normal
        n = [row['nx'], row['ny'], row['nz']]
        # get ransac normal
        rn = [row['rnx'], row['rny'], row['rnz']]
        # calculate angle
        n_angle = 90 - np.rad2deg(np.arccos(np.dot(gt_orientation, n)))
        rn_angle = 90 - np.rad2deg(np.arccos(np.dot(gt_orientation, rn)))
        # append to list
        cloud.at[idx, 'normal_dev_gt'] = abs(n_angle)
        cloud.at[idx, 'ransac_normal_dev_gt'] = abs(rn_angle)

    # plot both histograms
    fig = plt.figure()
    # fix bin widths for both histograms
    bins = np.linspace(0, 90, 91)
    plt.hist(cloud['normal_dev_gt'], bins=bins, alpha=0.5, label='normal')
    plt.hist(cloud['ransac_normal_dev_gt'], bins=bins, alpha=0.5, label='ransac normal')
    plt.xlabel('angle between normal and ground truth orientation')
    plt.ylabel('frequency')
    plt.legend()
    plt.show()

    # calculate mean and median deviation
    mean_dev = np.mean(cloud['normal_dev_gt'])
    median_dev = np.median(cloud['normal_dev_gt'])
    print(f'estimated normals vs. orientation gt: mean deviation: {mean_dev:.2f} degrees, median deviation: {median_dev:.2f} degrees')
    mean_dev = np.mean(cloud['ransac_normal_dev_gt'])
    median_dev = np.median(cloud['ransac_normal_dev_gt'])
    print(f'ransac normals vs. orientation gt:    mean deviation: {mean_dev:.2f} degrees, median deviation: {median_dev:.2f} degrees')
    a = 0






