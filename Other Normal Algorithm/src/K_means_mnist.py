import numpy as np
import time
import h5py
import torch
from evaluation import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.io import loadmat
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import jaccard_score as jaccard
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import fowlkes_mallows_score as fm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def load_mnist() -> tuple:
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.FashionMNIST(root='../data',
                                train=True,
                                download=True,
                                transform=tensor_transform)
    test_set = datasets.FashionMNIST(root='../data',
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


print("process MNIST dataset")
x_train, y_train, x_test, y_test = load_mnist()
y_train_nu = y_train.numpy()
print(x_train.shape)
x_train_flat = x_train.reshape(x_train.shape[0], -1)
count = check_count(y_train)
print(count)
# start_time = time.time()
# kmeans = KMeans(n_clusters=count, init='random', n_init=10, max_iter=1000, tol=0.001, algorithm='lloyd')
# y_pre = kmeans.fit_predict(x_train_flat)
# end_time = time.time()

# run_time = end_time - start_time

# acc_score = cluster_accuracy(y_train_nu, y_pre)
# nmi_score = nmi(y_train, y_pre)
# print(f"time: {np.round(run_time, 3)}")
# print(f"ACC: {np.round(acc_score, 3)}")
# print(f"NMI: {np.round(nmi_score, 3)}")
# print("begin to draw")
tsne = TSNE(n_components=2, init='random', random_state=501)
X_tsne = tsne.fit_transform(x_train_flat)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap='viridis')
#plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', label='Cluster Centers')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Original data of MNIST dataset')
plt.legend()
plt.colorbar()
plt.savefig("mnist_nor.png")