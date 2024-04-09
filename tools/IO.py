import os
import pathlib
import pickle

import numpy as np

from omegaconf import OmegaConf

def get_is_conf():
    config = OmegaConf.load('config.yaml')
    if os.name == 'nt':
        basepath = config.general.basepath_windows
    else:  # os.name == 'posix':
        basepath = config.general.basepath_macos
    config.general.path = pathlib.Path(f'{basepath}{config.general.project_path}{config.segmentation.cloud_path}')
    config.general.parking_path = pathlib.Path(f'{basepath}{config.general.project_path}{config.general.parking_path}')

    return config



def points2txt(pointset, path, topic):
    with open(f'{path}/{topic}.txt', 'w') as f:
        for i in range(pointset.shape[0]):
            f.write(f'{pointset[i][0]} {pointset[i][1]} {pointset[i][2]} \n')

    return


def cache_meta(data, path, topic):
    with open(f'{path}/{topic}.pickle', 'wb') as f:
        pickle.dump(data, f)

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

def cache_io(xyz=False, normals=False, supernormals=False, confidence=False,
             instance_gt=False, instance_pred=False, path=None, cloud=None, cache_flag=None):
    io_agenda = []
    if xyz:
        io_agenda.append('x')
        io_agenda.append('y')
        io_agenda.append('z')
    if normals:
        io_agenda.append('nx')
        io_agenda.append('ny')
        io_agenda.append('nz')
    if supernormals:
        io_agenda.append('supernormals')
    if confidence:
        io_agenda.append('confidence')
    if instance_gt:
        io_agenda.append('instance_gt')
    if instance_pred:
        io_agenda.append('instance_pred')

    cloud_to_write = cloud.loc[:, io_agenda]
    cloud_to_write.to_csv(f'{path}cache_cloud_{cache_flag}.txt', sep=' ', header=False, index=False)
