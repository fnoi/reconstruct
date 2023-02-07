import numpy as np
import matplotlib.pyplot as plt
import scipy

from sklearn.cluster import KMeans, DBSCAN

from tools.IO import points2txt

if __name__ == '__main__':
    with open('./data/in/pc_xyz33_reduced_2.txt', 'r') as f:
        data = f.readlines()
        data = [line.strip().split(' ') for line in data]
        data = np.array(data, dtype=np.float32)

    clustering = KMeans(n_clusters=4, random_state=0).fit(data)
    #clustering = DBSCAN(eps=0.1, min_samples=5).fit(data)
    labels = clustering.labels_

    no_cluster = len(np.unique(labels))
    print(f'Number of clusters: {no_cluster}')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='rainbow', s=0.1)
    axis_range = max([data[:, i].max() - data[:, i].min() for i in range(3)])

    xlim = (
        (np.max(data[:, 0]) + np.min(data[:, 0]))/2 - axis_range/2, (np.max(data[:, 0]) + np.min(data[:, 0]))/2 + axis_range/2)
    ylim = (
        (np.max(data[:, 1]) + np.min(data[:, 1]))/2 - axis_range/2, (np.max(data[:, 1]) + np.min(data[:, 1]))/2 + axis_range/2)
    zlim = (
        (np.max(data[:, 2]) + np.min(data[:, 2]))/2 - axis_range/2, (np.max(data[:, 2]) + np.min(data[:, 2]))/2 + axis_range/2)

    ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
    fig.show()

    a = 0
