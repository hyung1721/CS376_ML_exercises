from turtle import color
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.datasets import make_circles

def kmeans(data, k):
    num_data = np.shape(data)[0]

    cluster_mean = data[np.random.choice(range(num_data), k)]
    mean_dist_mat = np.linalg.norm(np.expand_dims(data, axis=0) - np.expand_dims(cluster_mean, axis=1), axis=2)
    cluster_allocation = np.argmin(mean_dist_mat, axis=0)

    updated = True
    init_num = 0

    while updated:
        updated = False

        cluster_mean = np.array([np.sum(data[cluster_allocation==i], axis = 0)/np.sum(cluster_allocation==i) for i in range(k)])
        mean_dist_mat = np.linalg.norm(np.expand_dims(data, axis=0) - np.expand_dims(cluster_mean, axis=1), axis=2)
        new_cluster_allocation = np.argmin(mean_dist_mat, axis=0)

        if not np.array_equal(cluster_allocation, new_cluster_allocation):
            updated = True
            cluster_allocation = new_cluster_allocation
        else:
            loss = np.sum(np.min(mean_dist_mat, axis= 0))
    
    return cluster_allocation, loss

def kernel_kmeans(data, k, s, kernel = 'gaussian'):
    num_data = np.shape(data)[0]

    cluster_mean = data[np.random.choice(range(num_data), k)]
    mean_dist_mat = np.linalg.norm(np.expand_dims(data, axis=0) - np.expand_dims(cluster_mean, axis=1), axis=2)
    cluster_allocation = np.argmin(mean_dist_mat, axis=0)

    if kernel == 'gaussian':
        pre_cal = np.exp(-np.square(metrics.pairwise_distances(data))/(2*(s**2)))
    else:
        print('NO such kernel')

    updated = True
    init_num = 0

    while updated:
        updated = False

        mean_dist_mat = [1 - 2*np.sum(pre_cal[:, cluster_allocation == idx], axis=1) / np.sum(cluster_allocation == idx)
                         + np.sum(pre_cal[cluster_allocation == idx][:, cluster_allocation == idx]) / (np.sum(cluster_allocation == idx)**2)
                         for idx in range(k)]
        new_cluster_allocation = np.argmin(mean_dist_mat, axis=0)

        if not np.array_equal(cluster_allocation, new_cluster_allocation):
            updated = True
            cluster_allocation = new_cluster_allocation
        else:
            loss = np.sum(np.min(mean_dist_mat, axis= 0))
    
    return cluster_allocation, loss

X, y = make_circles(n_samples=1000, noise=0.1, factor=0.3, random_state=10)
plt.figure()
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='black')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
plt.show()

np.random.seed(125452)
loss = np.inf

# for _ in range(10):
#     kmeans_result, loss_tmp = kmeans(X, k=2)
#     if loss > loss_tmp:
#         loss = loss_tmp
#         best_kmeans_result = kmeans_result
#         score = metrics.normalized_mutual_info_score(y, kmeans_result)

for _ in range(10):
    kkmeans_result, loss_tmp = kernel_kmeans(X, k=2, s=0.5)
    if loss > loss_tmp:
        loss = loss_tmp
        best_kkmeans_result = kkmeans_result
        score = metrics.normalized_mutual_info_score(y, kkmeans_result)


print(score)

plt.figure()
plt.scatter(X[kkmeans_result == 0, 0], X[kkmeans_result == 0, 1], color='black')
plt.scatter(X[kkmeans_result == 1, 0], X[kkmeans_result == 1, 1], color='blue')
plt.show()