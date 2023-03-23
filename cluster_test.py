from math import floor

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

if __name__ == "__main__":

    with open('C:/Users/ga25mal/PycharmProjects/reconstruct/data/test/test_beams_ok_enc_rgb.txt', 'r') as f:
        thisthat = np.loadtxt(f)

    # downsample every 20th point
    thisthat = thisthat[::100, :]
    coordinates = thisthat[:, :3]
    features = thisthat[:, 3:]
    scaler = StandardScaler()
    scaled_feat = scaler.fit_transform(features)
    data = np.hstack((coordinates, scaled_feat))

    eps_values = np.linspace(0.1, 10, 100)
    min_samples_values = [_ for _ in range(2, 100)]

    best_eps = None
    best_min_samples = None
    best_silhouette = -1

    result_frame = np.zeros((len(eps_values), len(min_samples_values)))

    finder = False
    if finder:

        for i, eps in enumerate(eps_values):
            for j, min_samples in enumerate(min_samples_values):
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(data)
                if len(np.unique(cluster_labels)) == 1:
                    silhouette = -1
                else:
                    silhouette = silhouette_score(data, cluster_labels)
                print(eps, min_samples, silhouette)
                result_frame[i, j] = silhouette
                if silhouette > best_silhouette:
                    best_eps = eps
                    best_min_samples = min_samples
                    best_silhouette = silhouette

        print(f'Best eps: {best_eps}')
        print(f'Best min_samples: {best_min_samples}')
        print(f'Best silhouette: {best_silhouette}')

        import matplotlib.pyplot as plt

        plt.imshow(result_frame, cmap='hot', interpolation='nearest')

    else:
        dbscan = DBSCAN(eps=2.5, min_samples=99)
        cluster_labels = dbscan.fit_predict(data)

        a = 0

    # encode cluster_labels in RGB

    # plot heatmap of results





    a = 0