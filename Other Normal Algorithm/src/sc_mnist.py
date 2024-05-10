import numpy as np
import time
import h5py
import torch
from evaluation import cluster_accuracy as acc1
from torchvision import datasets, transforms
from sklearn.cluster import SpectralClustering
from sklearn import preprocessing
from scipy.io import loadmat
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import jaccard_score as jaccard
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import fowlkes_mallows_score as fm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def load_mnist() -> tuple:
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root='../data',
                                train=True,
                                download=True,
                                transform=tensor_transform)
    test_set = datasets.MNIST(root='../data',
                                train=False,
                                download=True,
                                transform=tensor_transform)
    
    x_train, y_train = zip(*train_set)
    x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*test_set)
    x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)

    return x_train, y_train, x_test, y_test


def check_count(array):
    flat = array.flatten()
    unique = torch.unique(flat)
    return len(unique)


x_train, y_train, x_test, y_test = load_mnist()
y_train_nu = y_train.numpy()
x_train_flat = x_train.reshape(x_train.shape[0], -1)
start_time = time.time()
spectral_clustering = SpectralClustering(n_clusters=10, affinity='nearest_neighbors', n_neighbors=10, eigen_solver='lobpcg',random_state=42)
y = spectral_clustering.fit_predict(x_train_flat)
end_time = time.time()
run_time = end_time - start_time

acc_score = acc1(y_train_nu, y)
nmi_score = nmi(y_train, y)

print(f"time: {np.round(run_time, 3)}")
print(f"NMI: {np.round(nmi_score, 3)}")
print(f"ACC: {np.round(acc_score, 3)}")