import numpy as np

from structure.CloudSegment import CloudSegment

if __name__ == '__main__':

    cloud = 'data/point_cloud/beam_1.txt'
    pc = CloudSegment()
    pc.load_from_txt(cloud)

    pca = pc.calc_pca()

    # dext = pc.calc_pca2()

    pca_o3d = pc.calc_pca_o3d()

    # pc_ransac_cyl = pc.find_cylinder()

    point_on_plane = pc.pts2plane()

    a = 0