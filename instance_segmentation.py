import pathlib
import os
from omegaconf import OmegaConf

if __name__ == "__main__":
    pc_path = 'in_test/test_junction.txt'
    config = OmegaConf.load('config.yaml')
    # local runs only, add docker support
    if os.name == 'nt':
        path = 'C:/Users/ga25mal/'
    else:  # os.name == 'posix':
        path = '/Users/fnoic/'
    path = pathlib.Path(f'{path}PycharmProjects/reconstruct/data/{config.segmentation.cloud_path}')
    print(path)
    a = 0
