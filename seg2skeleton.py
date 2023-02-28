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

    skeleton = Skeleton(f'{str(os.getcwd())}/data/out/0_skeleton')
    if not os.path.exists(skeleton.path):
        os.makedirs(skeleton.path)

    segments: list = [f'beam_{i}' for i in range(1, 31)]

    for segment in segments:
        cloud = Segment(name=segment)
        cloud.load_from_txt(segment)
        cloud.calc_pca_o3d()

        cloud.plot_flats()

        cloud.transform_clean()
        cloud.pc2obj(pc_type='initial')
        skeleton.add(cloud)

    skeleton.find_joints()
    skeleton.join_passing()
    skeleton.join_on_passing()
    skeleton.to_obj(topic='intermediate')

