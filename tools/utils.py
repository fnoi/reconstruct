import numpy as np
import plotly.graph_objs as go

from tools.geometry import warped_vectors_intersection


def update_logbook_checklist(neighbors, skeleton, checklist):
    logbook = {}
    for neighbor in neighbors:
        logbook[neighbor] = warped_vectors_intersection(neighbor,
                                                        skeleton[neighbor[0]],
                                                        skeleton[neighbor[1]])
        if logbook[neighbor][3] == 0:
            checklist[neighbor[0]] += 1
        elif logbook[neighbor][3] == 1:
            checklist[neighbor[1]] += 1

    return logbook, checklist


def find_random_id(unavailable, all_ids):
    random_id = np.random.randint(0, len(all_ids))
    if random_id in unavailable:
        return find_random_id(unavailable, all_ids)
    else:
        return random_id


def calculate_miou(points_array):
    predicted_labels = points_array[:, 9]  # Predicted instance labels are in column 10 (index 9)
    ground_truth_labels = points_array[:, 11]  # Ground truth instance labels are in column 12 (index 11)

    unique_predicted = np.unique(predicted_labels)
    unique_ground_truth = np.unique(ground_truth_labels)

    iou_scores = []

    for pred_label in unique_predicted:
        if pred_label == 0:  # Assuming label 0 is background or invalid
            continue
        for gt_label in unique_ground_truth:
            if gt_label == 0:  # Assuming label 0 is background or invalid
                continue

            # Calculate Intersection and Union
            intersection = np.sum((predicted_labels == pred_label) & (ground_truth_labels == gt_label))
            union = np.sum((predicted_labels == pred_label) | (ground_truth_labels == gt_label))

            if union == 0:
                continue

            iou = intersection / union
            iou_scores.append(iou)

    # Compute mean IoU
    miou = np.mean(iou_scores) if iou_scores else 0
    return miou


# Example usage
# points_array = your_numpy_array_here
# miou = calculate_miou(points_array)
# print("Mean IoU:", miou)


import numpy as np





def calculate_miou_with_labels(points_array):
    predicted_labels = points_array[:, 5]  # Predicted instance labels are in column 10 (index 9)
    ground_truth_labels = points_array[:, 6]  # Ground truth instance labels are in column 12 (index 11)

    unique_predicted = np.unique(predicted_labels)
    unique_ground_truth = np.unique(ground_truth_labels)

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
            intersection = np.sum((predicted_labels == pred_label) & (ground_truth_labels == gt_label))
            union = np.sum((predicted_labels == pred_label) | (ground_truth_labels == gt_label))

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


def calculate_view_direction(roll, pitch, yaw):
    # Convert angles from degrees to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # Rotation matrices around the X, Y, and Z axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])

    Ry = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    Rz = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    # Default direction vector (assuming Z-axis points towards the viewer)
    direction = np.array([0, 0, 1])

    # Apply rotations
    direction_rotated = direction @ Rz @ Ry @ Rx

    return direction_rotated


def plot_patch(cloud_frame=None, seed_id=None, neighbor_ids=None):
    """plot local patch with selected aspects"""
    fig = go.Figure()
    # neighbor points scatter plot
    fig.add_trace(go.Scatter3d(
        x=cloud_frame.loc[neighbor_ids, 'x'],
        y=cloud_frame.loc[neighbor_ids, 'y'],
        z=cloud_frame.loc[neighbor_ids, 'z'],
        mode='markers',
        marker=dict(
            size=3,
            color='black',
            opacity=0.6
        )
    ))
    # seed point highlight scatter
    fig.add_trace(go.Scatter3d(
        x=[cloud_frame.loc[seed_id, 'x']],
        y=[cloud_frame.loc[seed_id, 'y']],
        z=[cloud_frame.loc[seed_id, 'z']],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            opacity=1
        )
    ))
    # normals for neighbors and seed with lines  # TODO: reduce length of normals
    len_fac_n = 0.2
    for i in neighbor_ids:
        fig.add_trace(go.Scatter3d(
            x=[cloud_frame.loc[i, 'x'], cloud_frame.loc[i, 'x'] + len_fac_n * cloud_frame.loc[i, 'nx']],
            y=[cloud_frame.loc[i, 'y'], cloud_frame.loc[i, 'y'] + len_fac_n * cloud_frame.loc[i, 'ny']],
            z=[cloud_frame.loc[i, 'z'], cloud_frame.loc[i, 'z'] + len_fac_n * cloud_frame.loc[i, 'nz']],
            mode='lines',
            line=dict(
                color='blue',
                width=0.5
            )
        ))
    # supernormal with thick orange line from seed  # TODO: reduce length of supernormal
    len_fac_sn = 0.25
    fig.add_trace(go.Scatter3d(
        x=[cloud_frame.loc[seed_id, 'x'], cloud_frame.loc[seed_id, 'x'] + len_fac_sn * cloud_frame.loc[seed_id, 'snx']],
        y=[cloud_frame.loc[seed_id, 'y'], cloud_frame.loc[seed_id, 'y'] + len_fac_sn * cloud_frame.loc[seed_id, 'sny']],
        z=[cloud_frame.loc[seed_id, 'z'], cloud_frame.loc[seed_id, 'z'] + len_fac_sn * cloud_frame.loc[seed_id, 'snz']],
        mode='lines',
        line=dict(
            color='orange',
            width=10
        )
    ))
    # header and layout, times new roman font
    fig.update_layout(
        title=f'local patch with seed point {seed_id}, {len(neighbor_ids)} neighbors, '
              f'confidence {cloud_frame.loc[seed_id, "confidence"]}',
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, showline=False, zeroline=False),
            yaxis=dict(showbackground=False, showgrid=False, showline=False, zeroline=False),
            zaxis=dict(showbackground=False, showgrid=False, showline=False, zeroline=False),
            bgcolor='white'
        ),
        font=dict(
            family='Times New Roman',
            size=18,
            color='black'
        )
    )
    fig.show()
