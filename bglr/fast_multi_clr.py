"""
FastmultiCLR: Anchor Selection + k-NN Bipartite Graph Construction

Uses K-means Nearest Points (KNP) for anchor selection with PyTorch-based
k-means (GPU accelerated when CUDA is available), and builds k-NN bipartite
graphs for multi-view data.
"""

import numpy as np
import torch
from .utils import l2_distance


def _pairwise_dist_sq(X, C):
    """Squared Euclidean distance: ||X[i] - C[j]||^2 for all i, j."""
    X_sq = (X ** 2).sum(dim=1, keepdim=True)   # (n, 1)
    C_sq = (C ** 2).sum(dim=1).unsqueeze(0)     # (1, k)
    cross = X @ C.T                              # (n, k)
    return X_sq + C_sq - 2 * cross               # (n, k)


def _kmeans_pytorch(X, n_clusters, n_iter=50, tol=1e-4, seed=42, n_init=3):
    """
    PyTorch-based k-means. Runs on GPU if CUDA is available, otherwise CPU.
    Uses multiple initializations and returns the best clustering.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input data.
    n_clusters : int
        Number of clusters.
    n_iter : int
        Maximum iterations per initialization.
    tol : float
        Convergence tolerance (relative centroid change).
    seed : int
        Random seed.
    n_init : int
        Number of initializations.

    Returns
    -------
    centroids : ndarray, shape (n_clusters, d)
    labels : ndarray, shape (n,)
    """
    rng = np.random.RandomState(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    n = X.shape[0]

    best_inertia = float('inf')
    best_centroids = None
    best_labels = None

    for _ in range(n_init):
        # k-means++ initialization
        idx0 = rng.randint(0, n)
        centroids = X_t[idx0:idx0 + 1].clone()  # (1, d)

        for _ in range(1, n_clusters):
            dist = _pairwise_dist_sq(X_t, centroids).min(dim=1)[0]  # (n,)
            probs = dist / dist.sum()
            cumprobs = probs.cumsum(0)
            r = torch.rand(1, device=device)
            new_idx = torch.searchsorted(cumprobs, r).item()
            new_idx = min(new_idx, n - 1)
            centroids = torch.cat([centroids, X_t[new_idx:new_idx + 1]], dim=0)

        prev_centroids = centroids.clone()

        for _ in range(n_iter):
            # Assign: compute squared distances and find nearest centroid
            dist_sq = _pairwise_dist_sq(X_t, centroids)    # (n, k)
            labels = dist_sq.argmin(dim=1)                  # (n,)

            # Update: vectorized centroid computation
            # one_hot: (n, k) -> normalize columns -> X^T @ one_hot
            one_hot = torch.zeros(n, n_clusters, dtype=torch.float32, device=device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            counts = one_hot.sum(dim=0).clamp(min=1)        # (k,)
            new_centroids = (X_t.T @ one_hot).T / counts.unsqueeze(1)  # (k, d)

            # Handle empty clusters
            empty_mask = one_hot.sum(dim=0) == 0
            if empty_mask.any():
                for j in empty_mask.nonzero(as_tuple=True)[0]:
                    new_centroids[j] = X_t[rng.randint(0, n)].clone()

            # Check convergence
            shift = (new_centroids - prev_centroids).norm(dim=1).max().item()
            prev_centroids = new_centroids.clone()
            centroids = new_centroids

            if shift < tol * (centroids.norm(dim=1).mean().item() + 1e-10):
                break

        # Compute inertia (sum of squared distances to nearest centroid)
        dist_sq = _pairwise_dist_sq(X_t, centroids)
        inertia = dist_sq.min(dim=1)[0].sum().item()

        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids.clone()
            best_labels = dist_sq.argmin(dim=1).clone()

    return best_centroids.cpu().numpy(), best_labels.cpu().numpy()


def fast_multi_clr(X_list, c, anchor_num, k=10):
    """
    For each view, select anchors via K-means Nearest Points (KNP) and
    build a k-NN bipartite graph connecting data points to anchors.
    K-means is implemented in PyTorch and runs on GPU when CUDA is available.

    Parameters
    ----------
    X_list : list of ndarray
        Each element X_list[v] is (n_samples, d_v), rows are samples.
    c : int
        Number of anchor clusters (k2).
    anchor_num : int
        Total number of anchors (= k2 * m).
    k : int
        Number of nearest anchors for bipartite graph (default 10).

    Returns
    -------
    centers : list of ndarray
        centers[v] is (anchor_num, d_v), anchor points for each view.
    B : list of ndarray
        B[v] is (n_samples, anchor_num), bipartite graph matrix for each view.
        Each row sums to 1.
    """
    n_view = len(X_list)
    n = X_list[0].shape[0]

    if k > anchor_num:
        k = anchor_num - 1

    # Concatenate all views for anchor selection
    XX = np.hstack([X_list[v] for v in range(n_view)]).astype(np.float32)  # (n, sum(d_v))

    m = anchor_num
    centers = [None] * n_view
    B = [None] * n_view

    # ---- K-means Nearest Points (KNP) Anchor Selection (GPU accelerated) ----
    km_centers, _ = _kmeans_pytorch(XX, m, n_iter=50, tol=1e-4, n_init=3)

    # Compute distances from each point to each cluster center on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    XX_t = torch.tensor(XX, dtype=torch.float32, device=device)
    cen_t = torch.tensor(km_centers, dtype=torch.float32, device=device)

    # dis[i,j] = ||XX[i] - cen[j]||^2
    XX_sq = (XX_t ** 2).sum(dim=1, keepdim=True)      # (n, 1)
    cen_sq = (cen_t ** 2).sum(dim=1).unsqueeze(0)      # (1, m)
    cross = XX_t @ cen_t.T                              # (n, m)
    dis = XX_sq + cen_sq - 2 * cross                    # (n, m)
    dis = dis.cpu().numpy()

    # For each cluster, pick the nearest data point as its anchor
    ind = np.argmin(dis, axis=0)
    ind = np.sort(ind)
    for v in range(n_view):
        centers[v] = X_list[v][ind, :]

    # ---- Build k-NN Bipartite Graphs for Each View (vectorized) ----
    for v in range(n_view):
        # Compute squared Euclidean distance between data and anchors
        D = l2_distance(X_list[v].T, centers[v].T)  # (n, m)

        # Sort distances: find k+1 nearest anchors per data point
        idx = np.argsort(D, axis=1)  # (n, m)

        # Vectorized k-NN weight computation
        # id_k: (n, k+1) — indices of k+1 nearest anchors per sample
        id_k = idx[:, :k + 1]
        # di: (n, k+1) — distances to those k+1 anchors
        di = np.take_along_axis(D, id_k, axis=1)

        # k-NN weight: w_i = (d_{k+1} - d_i) / (k * d_{k+1} - sum(d_{1:k}))
        denom = k * di[:, k] - di[:, :k].sum(axis=1)  # (n,)
        safe = np.abs(denom) > 1e-10
        weights = np.zeros((n, k))
        weights[safe] = (di[safe, k, None] - di[safe, :k]) / denom[safe, None]

        # Scatter weights into Bv
        Bv = np.zeros((n, m))
        row_idx = np.arange(n)[:, None]  # (n, 1)
        Bv[row_idx, id_k[:, :k]] = weights

        # Fallback: rows with near-zero weights → uniform
        zero_rows = Bv.sum(axis=1) < 1e-6
        if zero_rows.any():
            Bv[zero_rows] = 0.0
            zr_idx = np.where(zero_rows)[0]
            for i in zr_idx:
                Bv[i, id_k[i, :k]] = 1.0 / k

        # Row-normalize to sum to 1
        row_sum = Bv.sum(axis=1, keepdims=True)
        Bv = Bv / np.maximum(row_sum, 1e-14)
        B[v] = Bv

    return centers, B
