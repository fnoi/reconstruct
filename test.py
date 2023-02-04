import numpy as np

from structure.CloudSegment import CloudSegment

if __name__ == '__main__':

    segments: list = [
        'beam_1',
        'beam_2'
    ]

    for segment in segments:
        cloud = CloudSegment(name=segment)
        cloud.load_from_txt(segment)
        cloud.calc_pca_o3d()
        cloud.pts2plane()
