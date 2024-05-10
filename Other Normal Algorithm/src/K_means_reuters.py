import numpy as np
import time
import h5py
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from evaluation import cluster_accuracy as acc1
from sklearn import preprocessing
from scipy.io import loadmat
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import jaccard_score as jaccard
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import fowlkes_mallows_score as fm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def load_reuters() -> tuple:
    with h5py.File('../../spectralNet/data/Reuters/reutersidf_total.h5', 'r') as f:
        x = np.asarray(f.get('data'), dtype='float32')
        y = np.asarray(f.get('labels'), dtype='float32')

        n_train = int(0.9 * len(x))
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test

def check_count(array):
    flat = array.flatten()
    unique = torch.unique(flat)
    return len(unique)



print("begin to process data\n")
x_train, y_train, x_test, y_test = load_reuters()
y_train_nu = y_train.numpy()
print(x_train.shape)
# flat = y_train.flatten()
# unique = torch.unique(flat)
# print(len(unique))
count = check_count(y_train)
print(count)

print("begin to clustering by kmeans\n")
# tsne = TSNE(n_components=2, init='random', random_state=501)
# X_tsne = tsne.fit_transform(x_train)
# start_time = time.time()
# kmeans = KMeans(n_clusters=count, init='random', n_init=10,max_iter=100,tol=0.001)
# y_pre = kmeans.fit_predict(x_train)
# end_time = time.time()
# run_time = end_time - start_time

# acc_score = acc1(y_train_nu, y_pre)
# nmi_score = nmi(y_train, y_pre)
# print(f"time: {np.round(run_time, 3)}")
# print(f"ACC: {np.round(acc_score, 3)}")
# print(f"NMI: {np.round(nmi_score, 3)}")



# centers = kmeans.cluster_centers_

# # 绘制聚类结果
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
# plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', label='Cluster Centers')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.title('K-means Clustering with t-SNE Visualization')
# plt.legend()
# plt.colorbar()
# plt.savefig('reuters1.png')