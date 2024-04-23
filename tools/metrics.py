import numpy as np
from scipy.optimize import linear_sum_assignment


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
        intersection = np.sum((inst_pred == pred_label) & (inst_gt == gt_label))
        union = np.sum((inst_pred == pred_label) | (inst_gt == gt_label))

        if union == 0:
            continue

        iou = intersection / union
        iou_scores.append(iou)

    miou = np.mean(iou_scores) if iou_scores else 0

    return miou


def calculate_metrics(df_cloud, config):
    inst_pred = df_cloud['grown_patch'].to_numpy()
    inst_gt = df_cloud['instance_gt'].to_numpy()

    id_map = find_pairs(pred=inst_pred, gt=inst_gt, method='hungarian')
    miou = calculate_miou(inst_pred, inst_gt, id_map)
    print(f'hungarian miou:   {miou:.4f}')

    id_map = find_pairs(pred=inst_pred, gt=inst_gt, method='greedy_gt')
    miou = calculate_miou(inst_pred, inst_gt, id_map)
    print(f'greedy_gt miou:   {miou:.4f}')

    id_map = find_pairs(pred=inst_pred, gt=inst_gt, method='greedy_pred')
    miou = calculate_miou(inst_pred, inst_gt, id_map)
    print(f'greedy_pred miou: {miou:.4f}')

    inst_pred_unique = np.unique(inst_pred)
    inst_gt_unique = np.unique(inst_gt)

    id_map = find_pairs(df_cloud, method)

    unique_predicted = np.unique(inst_pred)
    unique_ground_truth = np.unique(inst_gt)

    iou_scores = []
    label_pairs = []  # Store label pairs with their IoU scores

    # For tracking best match and IoU for each predicted label
    best_matches = []

    for pred_label in unique_predicted:
        if pred_label == 0:  # Assuming label 0 is background or invalid
            continue
        best_iou = 0
        best_gt_label = None
        for gt_label in unique_ground_truth:
            # if gt_label == 0:  # Assuming label 0 is background or invalid
            #     continue

            # Calculate Intersection and Union
            intersection = np.sum((inst_pred == pred_label) & (inst_gt == gt_label))
            union = np.sum((inst_pred == pred_label) | (inst_gt == gt_label))

            if union == 0:
                continue

            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_gt_label = gt_label

        if best_gt_label is not None:
            iou_scores.append(best_iou)
            best_matches.append((pred_label, best_gt_label, best_iou))

    # Extract the label pairs and their IoUs for the best matches
    label_pairs = [(match[0], match[1]) for match in best_matches]

    # Compute mean IoU
    miou = np.mean(iou_scores) if iou_scores else 0

    return miou, label_pairs
