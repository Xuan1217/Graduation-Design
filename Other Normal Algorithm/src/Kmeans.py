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

location = "../data/uci_mat/D053.mat"
mat = loadmat(location)
data = mat['data']
label = np.array(mat['labels'])
labels = label.reshape(-1)
count = check_count(label)
print(count)
#print(labels)
# B=preprocessing.minmax_scale(B,axis=1)
# #kmeans参数设置
start_time = time.time()
kmeans=KMeans(n_clusters=count,init='random',n_init=100,max_iter=1000,tol=0.005, algorithm='auto')
#聚类
y = kmeans.fit_predict(data)
end_time = time.time()
run_time = end_time - start_time

acc_score = cluster_accuracy(labels, y)
nmi_score = nmi(labels, y)

print(f"time: {np.round(run_time, 3)}")
print(f"ACC: {np.round(acc_score, 3)}")
print(f"NMI: {np.round(nmi_score, 3)}")

#输出聚类中心坐标，行表示聚类中心，列表示聚类中心的分量
# print(f"cluster center : {kmeans.cluster_centers_}") 
#输出距离平方和
print(f"distance square sum : {kmeans.inertia_}")
# print("Silhouette Score:", silhouette_avg)
# #calinski_harabasz指数


# print("begin to draw")
# tsne = TSNE(n_components=2, init='random', random_state=501)
# X_tsne = tsne.fit_transform(data)

# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
# #plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', label='Cluster Centers')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.title('(Normal) MNIST Clustering with t-SNE Visualization')
# plt.legend()
# plt.colorbar()
# plt.savefig("d129_nor.png")