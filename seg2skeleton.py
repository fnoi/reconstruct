import itertools
import os
from math import ceil, sqrt
import matplotlib.pyplot as plt

import numpy as np

from structure.CloudSegment import Segment
from structure.SegmentSkeleton import Skeleton
from tools.geometry import warped_vectors_intersection, manipulate_skeleton
from tools.IO import lines2obj
from tools.utils import update_logbook_checklist

if __name__ == '__main__':

    skeleton = Skeleton(path=f'{str(os.getcwd())}/data/out/0_skeleton',
                        types=['pipes'])  # beams

    # pretty lost, import pipe data here according to convention
    if skeleton.pipes:
        with open(f'{str(os.getcwd())}/data/in_pipe/axis.txt', 'r') as f:
            data = f.readlines()
            data = [line.strip().split(' ') for line in data]
            pipe_ind = 0
            seg = Segment()
            for line in data:
                if int(line[0]) == pipe_ind:
                    skeleton.add_bone(line)


            data = np.array(data, dtype=np.float32)
            data = data[:, :3]

    if skeleton.beams:

        segments: list = [f'beam_{i}' for i in range(1, 31)]

        for segment in segments:
            cloud = Segment(name=segment)
            cloud.load_from_txt(segment)
            cloud.calc_pca_o3d()

            cloud.plot_flats()

            cloud.transform_clean()
            cloud.pc2obj(pc_type='initial')
            skeleton.add_cloud(cloud)

        skeleton.find_joints()
        skeleton.join_passing()
        skeleton.join_on_passing()
        skeleton.to_obj(topic='intermediate')
