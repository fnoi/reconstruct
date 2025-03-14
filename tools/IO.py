import os
import pathlib
import pickle
import ifcopenshell

import numpy as np
import pandas as pd

from omegaconf import OmegaConf

from tools.utils import calculate_view_direction


def config_io(config):
    if os.name == 'nt':
        config.project.path = pathlib.Path(f'{config.project.basepath_windows}{config.project.project_path}{config.data.cloud_path}')
        config.project.orientation_gt_path = pathlib.Path(f'{config.project.basepath_windows}{config.project.project_path}{config.data.orientation_path}')
    else:  # os.name == 'posix':
        config.project.path = pathlib.Path(f'{config.project.basepath_macos}{config.project.project_path}{config.data.cloud_path}')
        config.project.orientation_gt_path = pathlib.Path(f'{config.project.basepath_macos}{config.project.project_path}{config.data.orientation_path}')

    return config

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
    a = 0
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
             instance_gt=False, instance_pred=False, ransac_patch=False, ransac_normals=False,
             path=None, cloud=None, cache_flag=None):
    # serialize without structure loss pickle or json
    cloud.to_pickle(f'{path}cache_cloud_{cache_flag}.pickle')

    if not any([xyz, normals, supernormals, confidence, instance_gt, instance_pred, ransac_patch, ransac_normals]):
        return

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
        io_agenda.append('snx')
        io_agenda.append('sny')
        io_agenda.append('snz')
    if ransac_normals:
        io_agenda.append('rnx')
        io_agenda.append('rny')
        io_agenda.append('rnz')
    if confidence:
        io_agenda.append('confidence')
    if instance_gt:
        io_agenda.append('instance_gt')
    if instance_pred:
        io_agenda.append('instance_pred')
    if ransac_patch:
        io_agenda.append('ransac_patch')

    cloud_to_write = cloud.loc[:, io_agenda]
    cloud_to_write.to_csv(f'{path}cache_cloud_{cache_flag}.txt', sep=' ', header=False, index=False)

    return


def load_angles(yaml_path):
    orientation_gt = OmegaConf.load(yaml_path)
    vecs = {}
    for key in orientation_gt:
        rpy = orientation_gt[str(int(key))]
        rpy = [rpy.X1, rpy.Y2, rpy.Z3]
        gt_orientation = calculate_view_direction(rpy[0], rpy[1], rpy[2])
        vecs[int(key)] = gt_orientation
    return vecs


def data_from_IFC(path, direct=False):
    # load the ifc file
    profiles = ifcopenshell.open(path)
    ishapes = profiles.by_type("IfcIShapeProfileDef")
    if direct:
        return ishapes

    # extract beam params to dataframe
    beams = []
    for ishape in ishapes:
        beams.append([ishape.ProfileName, ishape.WebThickness, ishape.FlangeThickness, ishape.OverallWidth, ishape.OverallDepth])
    beam_df = pd.DataFrame(beams, columns=['name', 'tw', 'tf', 'bf', 'd'])
    # divide by 1000 to convert to meters, except for 'name'
    beam_df[['tw', 'tf', 'bf', 'd']] = beam_df[['tw', 'tf', 'bf', 'd']] / 1000
    beam_list = beam_df.values.tolist()
    beam_list = [beam[1:] for beam in beam_list]

    return beam_list, beam_df
