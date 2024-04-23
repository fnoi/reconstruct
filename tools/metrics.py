import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, precision_recall_curve


def find_pairs(pred, gt, method):
    match method:
        case 'greedy_gt':
            return find_pairs_greedy_gt(pred, gt)
        case 'greedy_pred':
            return find_pairs_greedy_pred(pred, gt)
        case 'hungarian':
            return find_pairs_hungarian(pred, gt)
        case _:
            raise ValueError(f"Invalid method: {method}")


def find_pairs_greedy_gt(pred, gt):
    """
    find the best matching between predicted and ground truth instances using a greedy algorithm
    """
    unique_gt, gt_counts = np.unique(gt, return_counts=True)
    unique_pred, pred_counts = np.unique(pred, return_counts=True)

    # sort uniques by count
    unique_gt = unique_gt[np.argsort(gt_counts)[::-1]]
    unique_pred = unique_pred[np.argsort(pred_counts)[::-1]]

    label_pairs = []
    for gt_label in unique_gt:
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
    unique_gt = unique_gt[np.argsort(gt_counts)[::-1]]
    unique_pred = unique_pred[np.argsort(pred_counts)[::-1]]

    label_pairs = []
    for pred_label in unique_pred:
        best_gt_label = None
        best_iou = 0
        for gt_label in unique_gt:
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
    find the best matching between predicted and ground truth instances using the Hungarian algorithm
    """
    unique_gt, gt_counts = np.unique(gt, return_counts=True)
    unique_pred, pred_counts = np.unique(pred, return_counts=True)

    num_gt_pad = max(0, len(unique_gt) - len(unique_pred))
    num_pred_pad = max(0, len(unique_pred) - len(unique_gt))

    # pad labels and ignore 0 (bc unlabeled)
    pred_labels = np.pad(unique_pred, (0, num_pred_pad), constant_values=0)
    gt_labels = np.pad(unique_gt, (0, num_gt_pad), constant_values=0)
    pred_counts = np.concatenate([pred_counts, np.zeros(num_pred_pad)])
    gt_counts = np.concatenate([gt_counts, np.zeros(num_gt_pad)])

    # create cost matrix with 1-iou as cost
    cost_matrix = np.zeros((len(pred_labels), len(gt_labels)))
    for i, pred_label in enumerate(pred_labels):
        for j, gt_label in enumerate(gt_labels):
            intersection = np.sum((pred == pred_label) & (gt == gt_label))
            union = np.sum((pred == pred_label) | (gt == gt_label))
            iou = intersection / union if union > 0 else 0
            cost_matrix[i, j] = 1 - iou

    # hungarian algorithm for minimum cost assignment (1-iou)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # return label pairs
    label_pairs = [(pred_labels[i], gt_labels[j]) for i, j in zip(row_ind, col_ind)]

    return label_pairs


def calculate_miou(inst_pred, inst_gt, id_map):
    iou_scores = []

    for pred_label, gt_label in id_map:
        if pred_label == 0 or gt_label == 0:
            continue
        intersection = np.sum((inst_pred == pred_label) & (inst_gt == gt_label))
        union = np.sum((inst_pred == pred_label) | (inst_gt == gt_label))

        if union == 0:
            continue

        iou = intersection / union
        iou_scores.append(iou)

    miou = np.mean(iou_scores) if iou_scores else 0

    return miou


def calculate_precision_recall(inst_pred, inst_gt, id_map, thresholds=None):
    inst_pred = -inst_pred
    for pred, gt in id_map:
        inst_pred[inst_pred == -pred] = gt
    inst_pred[inst_pred == 0] = -1
    inst_gt[inst_gt == 0] = -1

    pr_raw = {}
    for label in np.unique(inst_gt):
        if label == -1:
            continue
        tp = np.sum((inst_pred == label) & (inst_gt == label))
        fp = np.sum((inst_pred == label) & (inst_gt != label))
        tn = np.sum((inst_pred != label) & (inst_gt != label))
        fn = np.sum((inst_pred != label) & (inst_gt == label))

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0

        # count label in gt
        n_gt = np.sum(inst_gt == label)
        n_pred = np.sum(inst_pred == label)

        pr_raw[label] = (precision, recall, iou, n_gt, n_pred)

    if thresholds is None:
        thresholds = [0.0, 0.5]

    gt_total_valid = np.sum(inst_gt != -1)
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

        print(f"mAP   unweighted  @{threshold}: {pr_thresh[threshold]['mean_precision']:.4f}")
        print(f"mAP   weighted    @{threshold}: {pr_thresh[threshold]['mean_precision_weighted']:.4f}")
        print(f"mIoU  unweighted  @{threshold}: {pr_thresh[threshold]['mean_iou']:.4f}")
        print(f"mIoU  weighted    @{threshold}: {pr_thresh[threshold]['mean_iou_weighted']:.4f}")

    return pr_thresh


def calculate_metrics(df_cloud, config):
    inst_pred = df_cloud['grown_patch'].to_numpy()
    inst_gt = df_cloud['instance_gt'].to_numpy()

    print('hungarian matching')
    id_map = find_pairs(pred=inst_pred, gt=inst_gt, method='hungarian')
    miou = calculate_miou(inst_pred, inst_gt, id_map)
    map_dict = calculate_precision_recall(inst_pred, inst_gt, id_map)
    print(f'miou global:   {miou:.4f}')

    print('greedy_gt matching')
    id_map = find_pairs(pred=inst_pred, gt=inst_gt, method='greedy_gt')
    map_dict = calculate_precision_recall(inst_pred, inst_gt, id_map)
    miou = calculate_miou(inst_pred, inst_gt, id_map)
    print(f'miou global:   {miou:.4f}')

    print('greedy_pred matching')
    id_map = find_pairs(pred=inst_pred, gt=inst_gt, method='greedy_pred')
    map_dict = calculate_precision_recall(inst_pred, inst_gt, id_map)
    miou = calculate_miou(inst_pred, inst_gt, id_map)
    print(f'miou global: {miou:.4f}')


