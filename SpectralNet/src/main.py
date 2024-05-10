import sys
import json
import torch
import random
import time
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import *
from data import load_data
from metrics import Metrics
from sklearn.cluster import KMeans
from SpectralNet import SpectralNet
from scipy.spatial.distance import cdist


class InvalidMatrixException(Exception):
    pass


def set_seed(seed: int = 0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    config_path = sys.argv[1]
    with open (config_path, 'r') as f:
        config = json.load(f)

    dataset = config["dataset"]
    n_clusters = config["n_clusters"]
    should_check_generalization = config["should_check_generalization"]
    
    x_train, x_test, y_train, y_test = load_data(dataset)

    if not should_check_generalization:
        if y_train is None:
            x_train = torch.cat([x_train, x_test])
            
        else:
            x_train = torch.cat([x_train, x_test])
            y_train = torch.cat([y_train, y_test])
    
    try:
        spectralnet = SpectralNet(n_clusters=n_clusters, config=config)
        spectralnet.fit(x_train, y_train)

    except torch._C._LinAlgError:
        raise InvalidMatrixException("The output of the network is not a valid matrix to the orthogonalization layer. " 
                                     "Try to decrease the learning rate to fix the problem.") 

    if not should_check_generalization:

        cluster_assignments = spectralnet.predict(x_train) # return：标签，SpectralNet.py
        if y_train is not None:    
            y = y_train.detach().cpu().numpy()
            acc_score = Metrics.acc_score(cluster_assignments, y, n_clusters) # metrics.py
            nmi_score = Metrics.nmi_score(cluster_assignments, y)
            J_score = Metrics.Jaccard_index(y, cluster_assignments)
            ari_score = Metrics.ARI_score(y, cluster_assignments)
            fm_score = Metrics.FM_score(y, cluster_assignments)
            e_acc = Metrics.e_acc(y, cluster_assignments)
            #silhouette_avg = Metrics.sil_score(x_train, cluster_assignments)
            #calinski_harabasz_score_val = Metrics.c_h_score(x_train, cluster_assignments)
            #davies_bouldin_score_val = Metrics.d_b_score(x_train, cluster_assignments)
            embeddings = spectralnet.embeddings_
            print(f"ACC: {np.round(acc_score, 3)}")
            print(f"E_ACC: {np.round(e_acc, 3)}")
            print(f"NMI: {np.round(nmi_score, 3)}")
            print(f"Jaccard: {np.round(J_score, 3)}")
            print(f"ARI: {np.round(ari_score, 3)}")
            print(f"FM: {np.round(fm_score, 3)}")
            # print(f"silhouette: {np.round(silhouette_avg, 3)}")
            # print(f"calinski: {np.round(calinski_harabasz_score_val, 3)}")
            # print(f"davies_bouldin: {np.round(davies_bouldin_score_val, 3)}")

            return embeddings, cluster_assignments
        
    else:
        y_test = y_test.detach().cpu().numpy()
        spectralnet.predict(x_train) 
        train_embeddings = spectralnet.embeddings_
        test_assignments = spectralnet.predict(x_test)
        test_embeddings = spectralnet.embeddings_
        kmeans_train = KMeans(n_clusters=n_clusters).fit(train_embeddings) # 训练集使用神经网络嵌入后kmeans聚类
        dist_matrix = cdist(test_embeddings, kmeans_train.cluster_centers_) # 计算与质心的距离
        closest_cluster = np.argmin(dist_matrix, axis=1) # 取最小，分到相应的簇
        acc_score = Metrics.acc_score(closest_cluster, y_test, n_clusters)
        e_acc = Metrics.e_acc(y_test, closest_cluster)
        nmi_score = Metrics.nmi_score(closest_cluster, y_test)
        print(f"ACC: {np.round(acc_score, 3)}")
        print(f"NMI: {np.round(nmi_score, 3)}")
        print(f"E_ACC: {np.round(e_acc, 3)}")

        return test_embeddings, test_assignments



if __name__ == "__main__":
    start_time = time.time()
    embeddings, assignments = main()
    end_time = time.time()
    cost_time = end_time - start_time
    write_assignmets_to_file(assignments)
    print(f"The time cost: {np.round(cost_time, 3)} s")
    print("Your assignments were saved to the file 'cluster_assignments.csv!")
    print("begin to draw!")
    tsne = TSNE(n_components=2, init='random', random_state=501)
    X_tsne = tsne.fit_transform(embeddings)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=assignments, cmap='viridis')
    #plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', label='Cluster Centers')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Dimensionally reduced data of MNIST dataset')
    plt.legend()
    plt.colorbar()
    plt.savefig("mnist.png")
