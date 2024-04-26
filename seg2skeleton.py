import itertools
import os
from math import ceil, sqrt
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from structure.CloudSegment import Segment
from structure.SegmentSkeleton import Skeleton
from tools.geometry import warped_vectors_intersection, manipulate_skeleton, rotation_matrix_from_vectors
from tools.IO import lines2obj, cache_meta
from tools.utils import update_logbook_checklist
from tools.test_plots import plot_test_in, plot_test_out


def inst2skeleton(cloud_df, config, df_cloud_flag=False):
    # new approach has cloud with beams as dataframe. inject alternative from here
    skeleton = Skeleton(path=f'{str(os.getcwd())}/data/out_rev/skeleton',
                        types=['beams'])  # beams only

    if skeleton.beams:
        segments: list = [f'beam_{i}' for i in np.unique(cloud_df['instance_pr']) if i != 0]

        # segment_files: list = [_ for _ in os.listdir(f'{str(os.getcwd())}/data/in_beam/')]
        # segments: list = [_[:-4] for _ in segment_files]
        for segment in tqdm(segments, desc='loading segment data', total=len(segments)):
            cloud = Segment(name=segment)
            if df_cloud_flag:
                cloud.load_from_df(cloud_df, segment)
            else:
                cloud.load_from_txt(segment)

        for segment in tqdm(segments, desc='instance orientation and point projection (axes)', total=len(segments)):
            cloud.calc_axes()
            # cloud.calc_pca_o3d()

            # cloud.plot_flats()
            #
            # cloud.transform_clean()
            # cloud.pc2obj(pc_type='initial')
            # skeleton.add_cloud(cloud)
            a = 0

        a = 0

        skeleton.potential = np.array([0, 0, 0])
        # counter = 0
        # while np.sum(skeleton.potential) < 3:  # testing is 1 !
        #     print(f'Iteration {counter + 1}')
        #     skeleton.trim_passing()
        #     skeleton.join_passing_new()
        #     skeleton.join_on_passing()
        #     print(skeleton.potential)
        #     counter += 1
        #     if counter > 10:
        #         break

        # skeleton.join_passing_new()
        skeleton.join_passing()
        skeleton.join_on_passing()
        skeleton.trim_passing()

        # print('in 1')
        # skeleton.trim_passing()
        # print('in 2')
        # skeleton.join_passing_new()
        # print('in 3')
        # skeleton.join_on_passing()

        skeleton.to_obj(topic=f'store_beam')
        for bone in skeleton.bones:
            # bone.recompute_pca()
            x_vec = np.array([1, 0, 0])

            # here pcb_rot and pcc_rot needed?
            bone.rot_mat_pca = rotation_matrix_from_vectors(bone.pca, x_vec)
            cache_meta(data={'rot_mat_pca': bone.rot_mat_pca, 'rot_mat_pcb': bone.rot_mat_pcb},
                       path=bone.outpath, topic='rotations')


    a = 0

    # pretty lost, import pipe data here according to convention
    if skeleton.pipes:
        pth = f'{str(os.getcwd())}/data/in_pipe/axis.txt'
        # pth = f'{str(os.getcwd())}/data/test/test_x.txt'
        with open(pth, 'r') as f:
            test = False
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

            mean_radius = np.mean([skeleton_actual.bones[i].radius for i in range(len(skeleton_actual.bones))])
            for _ in skeleton_actual.bones:
                _.radius = mean_radius

            for i, skeleton in enumerate(skeletons):
                skeleton.potential = np.array([0, 0, 0])
                skeleton.to_obj(topic=f'skeletonraw_{i}')
                # counter = 0
                # while np.sum(skeleton.potential) < 3: #testing is 1 !
                #     print(f'Iteration {counter +1}')
                #     skeleton.trim_passing()
                #     skeleton.join_passing_new()
                #     skeleton.join_on_passing()
                #     print(skeleton.potential)
                #     counter += 1
                #     if counter > 10:
                #         break

                skeleton.join_passing_new()
                skeleton.join_on_passing()
                skeleton.trim_passing()

                # print('in 1')
                # skeleton.trim_passing()
                # print('in 2')
                # skeleton.join_passing_new()
                # print('in 3')
                # skeleton.join_on_passing()

                if test:
                    plot_test_out(skeleton, pth)
                skeleton.to_obj(topic=f'store_pipe{i}', radius=True)

                a = 0

    skeleton = Skeleton(path=f'{str(os.getcwd())}/data/out/0_skeleton',
                        types=['beams'])  # beams




if __name__ == '__main__':
    inst2skeleton()

