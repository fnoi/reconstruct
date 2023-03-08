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
from tools.test_plots import plot_test_in, plot_test_out

if __name__ == '__main__':

    # does skeleton need to be inside the loop actually?
    skeleton = Skeleton(path=f'{str(os.getcwd())}/data/out/0_skeleton',
                        types=['pipes'])  # beams

    # pretty lost, import pipe data here according to convention
    if skeleton.pipes:
        pth = f'{str(os.getcwd())}/data/test/test_2.txt'
        with open(pth, 'r') as f:
            test = True
        # with open(f'{str(os.getcwd())}/data/in_pipe/axis.txt', 'r') as f:
            data = f.readlines()
            data = [line.strip().split(' ') for line in data]
            if test:
                plot_test_in(data, pth)
            pipe_ind = -1
            skeletons = []
            for line in data:
                if int(line[0]) != pipe_ind:
                    skeleton_actual = Skeleton(path=f'{str(os.getcwd())}/data/out/0_skeleton', types=['pipes'])
                    skeletons.append(skeleton_actual)

                pipe_ind = int(line[0])
                seg = Segment(name=str(pipe_ind))

                seg.left = np.array([float(line[1]), float(line[2]), float(line[3])])
                seg.right = np.array([float(line[4]), float(line[5]), float(line[6])])
                seg.center = (seg.left + seg.right) / 2
                seg.radius = float(line[7])

                skeleton_actual.add_bone(seg)

            for i, skeleton in enumerate(skeletons):
                skeleton.potential = np.array([0, 0, 0])
                counter = 0
                while np.sum(skeleton.potential) < 1: #testing !
                    print(f'Iteration {counter +1}')
                    skeleton.join_passing_new()
                    # skeleton.trim_passing()
                    # skeleton.join_on_passing()
                    print(skeleton.potential)
                    counter += 1
                    if counter > 10:
                        break
                if test:
                    plot_test_out(skeleton, pth)
                skeleton.to_obj(topic=f'store_{i}')

                a = 0

    if skeleton.beams:

        segments: list = [f'beam_{i}' for i in range(1, 31)]
        # segment_files: list = [_ for _ in os.listdir(f'{str(os.getcwd())}/data/in_beam/')]
        # segments: list = [_[:-4] for _ in segment_files]
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
