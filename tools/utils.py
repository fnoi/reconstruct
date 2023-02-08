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