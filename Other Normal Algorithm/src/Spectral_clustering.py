from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import jaccard_score as jaccard
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import fowlkes_mallows_score as fm
from sklearn.metrics import accuracy_score as acc
from evaluation import cluster_accuracy as acc1
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn import preprocessing
from scipy.io import loadmat
import numpy as np
import time

# location = "../data/cl/data_batch_1.mat"
location = "../data/uci_mat/D053.mat"
mat = loadmat(location)
data = mat['data']
true_label = mat['labels']
true_label = true_label.reshape(-1)

# data = preprocessing.minmax_scale(data,axis=1)
start_time = time.time()
spectral_clustering = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=10, eigen_solver='lobpcg',random_state=42)
y = spectral_clustering.fit_predict(data)
end_time = time.time()
run_time = end_time - start_time

acc_score = acc1(true_label, y)
nmi_score = nmi(true_label, y)

print(f"time: {np.round(run_time, 3)}")
print(f"NMI: {np.round(nmi_score, 3)}")
print(f"ACC: {np.round(acc_score, 3)}")
