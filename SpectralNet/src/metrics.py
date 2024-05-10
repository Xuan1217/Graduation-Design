import numpy as np
import sklearn.metrics as metrics

from evaluation import cluster_accuracy as acc1
from utils import *
from munkres import Munkres
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import jaccard_score as jaccard
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import fowlkes_mallows_score as fm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class Metrics:
    @staticmethod
    def acc_score(cluster_assignments: np.ndarray, y: np.ndarray, n_clusters: int)  -> float:
        """
        Computes the accuracy score of the clustering algorithm
        Args:
            cluster_assignments (np.ndarray):   cluster assignments for each data point
            y (np.ndarray):                     ground truth labels
            n_clusters (int):                   number of clusters

        Returns:
            float: accuracy score
        """

        confusion_matrix = metrics.confusion_matrix(y, cluster_assignments, labels=None)
        cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters=n_clusters)
        indices = Munkres().compute(cost_matrix)
        kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
        y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
        print(metrics.confusion_matrix(y, y_pred))
        accuracy = np.mean(y_pred == y)
        return accuracy
    
    @staticmethod
    def nmi_score(cluster_assignments: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the normalized mutual information score of the clustering algorithm
        Args:
            cluster_assignments (np.ndarray):   cluster assignments for each data point
            y (np.ndarray):                     ground truth labels

        Returns:
            float: normalized mutual information score
        """
        return nmi(cluster_assignments, y)

    @staticmethod
    def Jaccard_index(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
        return jaccard(true_labels, cluster_labels, average='macro')

    @staticmethod
    def ARI_score(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
        return ari(true_labels, cluster_labels)

    @staticmethod
    def FM_score(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
        return fm(true_labels, cluster_labels)
    
    @staticmethod
    def sil_score(origin_data: np.ndarray, cluster_labels: np.ndarray) -> float:
        return silhouette_score(origin_data, cluster_labels)
    
    @staticmethod
    def c_h_score(origin_data: np.ndarray, cluster_labels: np.ndarray) -> float:
        return calinski_harabasz_score(origin_data, cluster_labels)
    
    @staticmethod
    def d_b_score(origin_data: np.ndarray, cluster_labels: np.ndarray) -> float:
        return davies_bouldin_score(origin_data, cluster_labels)
    
    @staticmethod
    def e_acc(origin_data: np.ndarray, cluster_labels: np.ndarray)  -> float:
        return acc1(origin_data, cluster_labels)