import numpy as np
import time
from evaluation import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.io import loadmat
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import jaccard_score as jaccard
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import fowlkes_mallows_score as fm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# default_char1 = "../data/ucl_mat/D00"
# default_char3 = "../data/ucl_mat/D0"
# default_char4 = "../data/ucl_mat/D"
# default_char2 = ".mat"


def check_count(array):
    return len(set(tuple(row) for row in array))

location = "../data/uci_mat/D153.mat"
mat = loadmat(location)
data = mat['data']
label = np.array(mat['labels'])
labels = label.reshape(-1)
count = check_count(label)
print(count)
print(data.shape)
print("begin to draw")
tsne = TSNE(n_components=2, init='random', random_state=501)
X_tsne = tsne.fit_transform(data)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
#plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', label='Cluster Centers')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2 ')
plt.title("Original data of Seeds dataset")
plt.legend()
plt.colorbar()
plt.savefig("pic/D153.png")