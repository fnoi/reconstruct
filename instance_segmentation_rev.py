import os
import pathlib

import pandas as pd
from omegaconf import OmegaConf

if __name__ == '__main__':
    config = OmegaConf.load('config_rev.yaml')
    if os.name == 'nt':
        config.project.path = pathlib.Path(f'{config.project.basepath_windows}{config.project.project_path}{config.segmentation.cloud_path}')
    else:  # os.name == 'posix':
        config.project.path = pathlib.Path(f'{config.project.basepath_macos}{config.project.project_path}{config.segmentation.cloud_path}')

    cache_flag = 0  # 0: no cache, 1: load normals, 2: load supernormals and confidence

    if cache_flag == 0: # no cache
        print('\n- No cache')
        with open(config.project.path, 'r') as f:
            # TODO: add option to load rgb here, currently XYZ, label only
            cloud = pd.read_csv(f, sep=' ', header=None).values
            cloud = pd.DataFrame(cloud, columns=['x', 'y', 'z', 'r', 'g', 'b', 'instance'])
            cloud.drop(['r', 'g', 'b'], axis=1, inplace=True)
        del f


    a = 0
