import numpy as np

from structure.PointCloud import PointCloud

if __name__ == '__main__':

    pc = PointCloud()
    pc.load_from_txt('data/point_cloud/pipe_segment_1.txt')

    pca = pc.calc_pca()

    # dext = pc.calc_pca2()

    pca_o3d = pc.calc_pca_o3d()

    pc_ransac_cyl = pc.find_cylinder()

    a = 0