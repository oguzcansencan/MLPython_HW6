from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split

n_samples_1 = 1000
n_samples_2 = 100
centers = [[0.0, 0.0], [2.0, 2.0]]
clusters_std = [1.5, 0.5]
X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers, cluster_std=clusters_std, random_state=0, shuffle=False)

X_circles, y_circles = make_circles(500, factor=0.1, noise=0.1)