from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split

#####################################################################
# First Input

n_samples_1 = 1000
n_samples_2 = 100
centers = [[0.0, 0.0], [2.0, 2.0]]
clusters_std = [1.5, 0.5]
X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers, cluster_std=clusters_std, random_state=0, shuffle=False)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2


#####################################################################
# Second Input

X_circles, y_circles = make_circles(500, factor=0.1, noise=0.1)

X_temp_circles, X_test_circles, y_temp_circles, y_test_circles = train_test_split(X_circles, y_circles, test_size=0.2, random_state=42)
X_train_circles, X_val_circles, y_train_circles, y_val_circles = train_test_split(X_temp_circles, y_temp_circles, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
