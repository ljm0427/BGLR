"""
Clustering evaluation metrics: NMI, ACC (Hungarian label mapping), purity,
F-score, precision, recall and silhouette.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
from .utils import kmeans


def best_map(L1, L2):
    """
    Permute labels L2 to match L1 via Hungarian algorithm (vectorized).

    Parameters
    ----------
    L1 : ndarray (n,), ground truth labels
    L2 : ndarray (n,), predicted labels

    Returns
    -------
    newL2 : ndarray (n,), permuted labels
    """
    Label1 = np.unique(L1)
    Label2 = np.unique(L2)
    nClass1 = len(Label1)
    nClass2 = len(Label2)

    # Vectorized contingency matrix via index mapping
    _, inv1 = np.unique(L1, return_inverse=True)
    _, inv2 = np.unique(L2, return_inverse=True)
    G = np.zeros((nClass2, nClass1), dtype=int)
    np.add.at(G, (inv2, inv1), 1)

    # Hungarian: row = L2 index, col = L1 index
    row_ind, col_ind = linear_sum_assignment(-G)

    # Build mapping vectorized
    map_L2_to_L1 = np.zeros(nClass2, dtype=int)
    for i in range(nClass2):
        if i in row_ind:
            pos = np.where(row_ind == i)[0][0]
            map_L2_to_L1[i] = col_ind[pos]
        else:
            map_L2_to_L1[i] = np.argmax(G[i, :])

    # Vectorized label permutation
    newL2 = Label1[map_L2_to_L1[inv2]]

    return newL2


def clustering_accuracy(true_labels, pred_labels):
    """
    Compute clustering accuracy via Hungarian matching.

    Parameters
    ----------
    true_labels : ndarray (n,)
    pred_labels : ndarray (n,)

    Returns
    -------
    acc : float
    """
    new_labels = best_map(true_labels, pred_labels)
    acc = np.mean(true_labels == new_labels)
    return acc


def compute_f(true_labels, pred_labels):
    """
    Compute F-score, Precision, Recall via contingency table (O(n + k^2)).
    Avoids building O(n^2) pairwise matrices.

    Parameters
    ----------
    true_labels : ndarray (n,)
    pred_labels : ndarray (n,)

    Returns
    -------
    fscore : float
    precision : float
    recall : float
    """
    true_labels = true_labels.reshape(-1)
    pred_labels = pred_labels.reshape(-1)

    # Build contingency table: C[i,j] = count of cluster i in true, cluster j in pred
    ut, inv_t = np.unique(true_labels, return_inverse=True)
    up, inv_p = np.unique(pred_labels, return_inverse=True)
    C = np.zeros((len(ut), len(up)), dtype=np.int64)
    np.add.at(C, (inv_t, inv_p), 1)

    # TP = sum_{i,j} C[i,j] choose 2
    # FP = sum_j (C[:,j].sum choose 2) - TP
    # FN = sum_i (C[i,:].sum choose 2) - TP
    n_ij = C.astype(np.float64)
    tp = np.sum(n_ij * (n_ij - 1)) / 2

    col_sums = C.sum(axis=0).astype(np.float64)
    row_sums = C.sum(axis=1).astype(np.float64)
    fp = np.sum(col_sums * (col_sums - 1)) / 2 - tp
    fn = np.sum(row_sums * (row_sums - 1)) / 2 - tp

    precision = tp / max(tp + fp, 1e-10)
    recall = tp / max(tp + fn, 1e-10)
    fscore = 2 * precision * recall / max(precision + recall, 1e-10)

    return fscore, precision, recall


def purity(true_labels, pred_labels):
    """
    Compute clustering purity via vectorized contingency table.

    Parameters
    ----------
    true_labels : ndarray (n,)
    pred_labels : ndarray (n,)

    Returns
    -------
    pur : float
    """
    _, inv_t = np.unique(true_labels, return_inverse=True)
    _, inv_p = np.unique(pred_labels, return_inverse=True)
    C = np.zeros((inv_t.max() + 1, inv_p.max() + 1), dtype=int)
    np.add.at(C, (inv_t, inv_p), 1)
    pur = np.sum(C.max(axis=1)) / len(true_labels)
    return pur


def kmeans_clustering(U, num_class, labels_init, max_iter=100):
    """
    L2-normalize the rows, then run K-means.

    Parameters
    ----------
    U : ndarray (n, d), each row is a sample
    num_class : int
    labels_init : ndarray (n,)
    max_iter : int

    Returns
    -------
    labels : ndarray (n,)
    v_obj_values : ndarray
    """
    # L2 normalize each row
    norm_U = np.sqrt(np.sum(U ** 2, axis=1, keepdims=True))
    U_normalized = U / np.maximum(norm_U, 1e-14)

    # K-means operates on columns as samples: (d, n)
    data = U_normalized.T
    v_obj_values, labels, _ = kmeans(data, num_class, labels_init, max_iter)

    return labels, v_obj_values


def silhouette_score(X, labels):
    """
    Exact Euclidean silhouette coefficient of a partition.

    Thin wrapper over :func:`sklearn.metrics.silhouette_score` so that the
    in-core diagnostic and the label-free model-selection score of
    ``run_grid_search.py`` are produced by exactly the same routine.

    Parameters
    ----------
    X : ndarray (n, d), data points (each row is a sample)
    labels : ndarray (n,), cluster assignments

    Returns
    -------
    score : float, mean silhouette coefficient (0.0 if fewer than 2 clusters)
    """
    from sklearn.metrics import silhouette_score as _sklearn_silhouette

    labels = np.asarray(labels).reshape(-1)
    if labels.size <= 1 or len(np.unique(labels)) <= 1:
        return 0.0
    return float(_sklearn_silhouette(X, labels, metric='euclidean'))


def clustering_measure(Q, true_labels, times_clustering=10, seed=42,
                       return_labels=False):
    """
    Run K-means multiple times on the learned representation Q,
    pick the best (lowest objective), compute all metrics.

    Parameters
    ----------
    Q : ndarray (n, d), learned representation (each row is a sample)
    true_labels : ndarray (n,), ground truth labels
    times_clustering : int, number of K-means runs
    seed : int, random seed
    return_labels : bool
        When True, the predicted labels behind the reported metrics are
        appended to the returned tuple (default False, so existing callers and
        the exported signature keep working).

    Returns
    -------
    nmi : float
    acc : float
    purity_val : float
    fscore : float
    precision : float
    labels : ndarray (n,), optional
        Appended only when ``return_labels=True``.
    """
    rng = np.random.RandomState(seed)
    n = Q.shape[0]
    cluster_num = len(np.unique(true_labels))

    # Generate random initial labels
    init_labels = np.ceil(cluster_num * rng.rand(n, times_clustering)).astype(int)

    best_obj = np.inf
    best_labels = None

    for i in range(times_clustering):
        labels, obj_values = kmeans_clustering(Q, cluster_num, init_labels[:, i])
        # obj_values[0] is the initial objective
        obj = obj_values[0]
        if obj < best_obj:
            best_obj = obj
            best_labels = labels.copy()

    # Compute metrics
    nmi = normalized_mutual_info_score(true_labels, best_labels)
    acc = clustering_accuracy(true_labels, best_labels)
    purity_val = purity(true_labels, best_labels)
    fscore, precision, _ = compute_f(true_labels, best_labels)

    if return_labels:
        return nmi, acc, purity_val, fscore, precision, best_labels
    return nmi, acc, purity_val, fscore, precision
