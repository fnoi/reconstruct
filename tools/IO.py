import os

import numpy as np


def points2txt(pointset, path, topic):
    with open(f'{path}/{topic}.txt', 'w') as f:
        for i in range(pointset.shape[0]):
            f.write(f'{pointset[i][0]} {pointset[i][1]} {pointset[i][2]} \n')

    return


def lines2obj(lines, path=os.getcwd(), topic='None', center=np.array([0.0, 0.0, 0.0])):
    a= 0
    if len(lines) == 1:
        with open(f'{path}/{topic}.obj', 'w') as f:
            f.write(f'v {lines[0][0][0]} {lines[0][0][1]} {lines[0][0][2]} \n'
                    f'v {lines[0][1][0]} {lines[0][1][1]} {lines[0][1][2]} \n'
                    f'l 1 2 \n')
    elif len(lines) == 2:
        pcab = np.stack(lines, axis=0)
        with open(f'{path}/{topic}.obj', 'w') as f:
            f.write(f'v {center[0]} {center[1]} {center[2]} \n'
                    f'v {pcab[0][0] + center[0]} {pcab[0][1] + center[1]} {pcab[0][2] + center[2]} \n'
                    f'v {pcab[1][0] + center[0]} {pcab[1][1] + center[1]} {pcab[1][2] + center[2]} \n'
                    f'l 1 2 \n'
                    f'l 1 3 \n')
    elif len(lines) == 3:
        pcab = np.stack(lines, axis=0)
        with open(f'{path}/{topic}.obj', 'w') as f:
            f.write(f'v {center[0]} {center[1]} {center[2]} \n'
                    f'v {pcab[0][0] + center[0]} {pcab[0][1] + center[1]} {pcab[0][2] + center[2]} \n'
                    f'v {pcab[1][0] + center[0]} {pcab[1][1] + center[1]} {pcab[1][2] + center[2]} \n'
                    f'v {pcab[2][0] + center[0]} {pcab[2][1] + center[1]} {pcab[2][2] + center[2]} \n'
                    f'l 1 2 \n'
                    f'l 1 3 \n'
                    f'l 1 4')
    else:
        raise ValueError('lines must be a list of length 1, 2 or 3')

    return
