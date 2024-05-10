import numpy as np
import time
import h5py
import torch
from sklearn.cluster import SpectralClustering
from sklearn import preprocessing
from scipy.io import loadmat
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import jaccard_score as jaccard
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import fowlkes_mallows_score as fm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def load_reuters() -> tuple:
    with h5py.File('../data/Reuters/reutersidf_total.h5', 'r') as f:
        x = np.asarray(f.get('data'), dtype='float32')
        y = np.asarray(f.get('labels'), dtype='float32')

        n_train = int(0.2 * len(x))
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test

print("begin to process data")
x_train, y_train, x_test, y_test = load_reuters()
# x_train = preprocessing.minmax_scale(x_train,axis=1)
print("begin to clustering")
start_time = time.time()
spectral_clustering = SpectralClustering(n_clusters=5, random_state=42)
y_pre = spectral_clustering.fit_predict(x_train)
end_time = time.time()
run_time = end_time - start_time

acc_score = acc(y_train, y_pre)
nmi_score = nmi(y_train, y_pre)
print(f"time: {np.round(run_time, 3)}")
print(f"ACC: {np.round(acc_score, 3)}")
print(f"NMI: {np.round(nmi_score, 3)}")