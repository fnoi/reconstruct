import itertools
import os
import pickle
from math import ceil, sqrt
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from structure.Cloud import Segment
from structure.Skeleton import Skeleton
from tools.geometry import warped_vectors_intersection, manipulate_skeleton, rotation_matrix_from_vectors
from tools.IO import lines2obj, cache_meta
from tools.utils import update_logbook_checklist
from tools.test_plots import plot_test_in, plot_test_out


def inst2skeleton(cloud_df, config, df_cloud_flag=False, plot=True):
    # new approach has cloud with beams as dataframe. inject alternative from here
    skeleton = Skeleton(path=f'{str(os.getcwd())}/data/out_rev/skeleton',
                        types=['beams'],
                        config=config)  # beams only

    if skeleton.beams:
        segments: list = [f'beam_{i}' for i in np.unique(cloud_df['instance_pr']) if i != 0]
        for segment in segments:
            cloud = Segment(name=segment, config=config)
            if df_cloud_flag:
                cloud.load_from_df(cloud_df, segment)
            else:
                cloud.load_from_txt(segment)

            if len(cloud.points) > config.skeleton.min_points:
                print(f'in for segment {segment} with initial size {len(cloud.points)}')
                cloud.calc_axes(plot=plot)
                if not cloud.break_flag:
                    skeleton.add_cloud(cloud)
                else:
                    print(f'dumping segment {segment} due to break flag')
            else:
                print(f'dumping segment {segment} due to initial size')

        return skeleton


if __name__ == '__main__':
    inst2skeleton()

