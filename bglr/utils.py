"""
Shared numerical helpers: feature normalization, squared Euclidean distances,
simplex projection and k-means.
"""

import numpy as np
import torch
from scipy import sparse


def normalize_fea(fea, row=1):
    """
    Normalize each row (row=1) or column (row=0) to unit L2 norm.

    Parameters
    ----------
    fea : ndarray (n_samples, n_features) or (n_features, n_samples)
    row : int
        1 = normalize rows, 0 = normalize columns

    Returns
    -------
    fea : ndarray, same shape, L2-normalized
    """
    if row:
        # Normalize each row: fea = diag(1/norm) @ fea
        n_smp = fea.shape[0]
        fea_norm = np.maximum(1e-14, np.sum(fea ** 2, axis=1))
        fea = sparse.spdiags(fea_norm ** -0.5, 0, n_smp, n_smp) @ fea
        if sparse.issparse(fea):
            fea = fea.toarray()
        else:
            fea = np.asarray(fea)
    else:
        # Normalize each column: fea = fea @ diag(1/norm)
        n_smp = fea.shape[1]
        fea_norm = np.maximum(1e-14, np.sum(fea ** 2, axis=0))
        if sparse.issparse(fea):
            fea = fea @ sparse.spdiags(fea_norm ** -0.5, 0, n_smp, n_smp)
            fea = fea.toarray()
        else:
            fea = fea @ np.diag(fea_norm ** -0.5)

    return fea


def l2_distance(a, b):
    """
    Compute squared Euclidean distance: ||A - B||^2 = ||A||^2 + ||B||^2 - 2*A'*B

    Parameters
    ----------
    a : ndarray (d, n1), each column is a data point
    b : ndarray (d, n2), each column is a data point

    Returns
    -------
    d : ndarray (n1, n2), squared Euclidean distance matrix
    """
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
        b = np.vstack([b, np.zeros((1, b.shape[1]))])

    aa = np.sum(a * a, axis=0)        # (n1,)
    bb = np.sum(b * b, axis=0)        # (n2,)
    ab = a.T @ b                       # (n1, n2)

    d = np.tile(aa[:, np.newaxis], (1, bb.shape[0])) + np.tile(bb, (aa.shape[0], 1)) - 2 * ab
    d = np.real(d)
    d = np.maximum(d, 0)
    return d


def opt_s(v, k=1):
    """
    Vectorized simplex projection.
    Solve: min 1/2 ||x - v||^2  s.t. x >= 0, sum(x) = k

    Uses a sort-based O(d log d) algorithm per column, fully vectorized in NumPy.

    Parameters
    ----------
    v : ndarray (d,) or (d, n)
    k : float, sum constraint (default 1)

    Returns
    -------
    x : ndarray, projected onto probability simplex (scaled by k)
    ft : int, always 0 (vectorized, no Newton iterations)
    """
    v = np.asarray(v, dtype=np.float64)
    single_col = (v.ndim == 1)
    if single_col:
        v = v.reshape(-1, 1)
    d, n = v.shape
    
    # Sort each column in descending order: (d, n)
    u = np.sort(v, axis=0)[::-1, :]
    # Cumulative sum along rows: cssv[j, :] = sum_{i=0..j} u[i, :]
    cssv = np.cumsum(u, axis=0)  # (d, n)
    
    # Find rho for each column: max j where u_j > (cssv_j - k) / j
    # Equivalent: j * u_j > cssv_j - k  →  j * u_j - cssv_j + k > 0
    j_range = np.arange(1, d + 1, dtype=np.float64).reshape(-1, 1)  # (d, 1)
    cond = j_range * u - cssv + k > 1e-12  # (d, n), bool
    
    # rho = argmax of cond along axis 0 (last True index)
    # Use: rho = d - argmax(cond[::-1]) - 1, but simpler: argmax on reversed
    cond_rev = cond[::-1, :]  # reverse rows
    rho_idx = d - 1 - np.argmax(cond_rev, axis=0)  # (n,)
    # If no True, argmax returns 0 → rho_idx = d-1, but should be -1
    no_true = ~np.any(cond, axis=0)
    rho_idx[no_true] = -1
    
    # Compute theta
    theta = np.zeros(n)
    mask = rho_idx >= 0
    if mask.any():
        # cssv[rho_idx[mask], mask] → use fancy indexing
        idx_r = rho_idx[mask]
        cssv_selected = cssv[idx_r, np.arange(n)[mask]]
        theta[mask] = (cssv_selected - k) / (idx_r + 1)
    
    x = np.maximum(v - theta.reshape(1, -1), 0)
    
    if single_col:
        return x.ravel(), 0
    return x, 0


def opt_s_torch(v, k=1.0):
    """
    GPU/CPU torch version of opt_s — simplex projection.
    Solve: min 1/2 ||x - v||^2  s.t. x >= 0, sum(x) = k

    Vectorized sort-based O(d log d) algorithm, runs entirely on GPU.
    No CPU-GPU data transfer needed.

    Parameters
    ----------
    v : torch.Tensor (d,) or (d, n)
    k : float, sum constraint (default 1)

    Returns
    -------
    x : torch.Tensor, projected onto probability simplex (scaled by k)
    """
    single_col = (v.ndim == 1)
    if single_col:
        v = v.unsqueeze(1)  # (d, 1)
    d, n = v.shape
    device = v.device

    # Sort each column in descending order
    u, _ = torch.sort(v, dim=0, descending=True)  # (d, n)
    cssv = torch.cumsum(u, dim=0)  # (d, n)

    # Find rho for each column: max j where j*u_j - cssv_j + k > 0
    j_range = torch.arange(1, d + 1, dtype=v.dtype, device=device).unsqueeze(1)  # (d, 1)
    cond = j_range * u - cssv + k > 1e-12  # (d, n)

    # Find last True per column (argmax on reversed)
    cond_rev = cond.flip(0)
    rho = d - 1 - torch.argmax(cond_rev.to(torch.int64), dim=0)  # (n,)
    no_true = ~cond.any(dim=0)
    rho[no_true] = -1

    # Compute theta
    theta = torch.zeros(n, dtype=v.dtype, device=device)
    mask = rho >= 0
    if mask.any():
        idx_r = rho[mask].long()
        cssv_selected = cssv[idx_r, torch.arange(n, device=device)[mask]]
        theta[mask] = (cssv_selected - k) / (idx_r.float() + 1)

    x = torch.clamp(v - theta.unsqueeze(0), min=0)

    if single_col:
        return x.squeeze(1)
    return x


def kmeans(data, K, labels_init, tmax=100):
    """
    Standard K-means with a given initialization.

    Parameters
    ----------
    data : ndarray (d, n), each column is a sample
    K : int, number of clusters
    labels_init : ndarray (n,), initial labels (1-indexed or 0-indexed)
    tmax : int, max iterations

    Returns
    -------
    v_obj_values : ndarray (tmax,)
    labels : ndarray (n,), cluster assignments (1-indexed)
    P : ndarray (d, K), cluster centers
    """
    d, n = data.shape
    v_obj_values = np.zeros(tmax)

    # Ensure labels are 0-indexed internally
    labels = labels_init.copy()
    if labels.min() >= 1:
        labels = labels - 1
    labels = labels.astype(int)

    # One-hot encoding
    U = np.zeros((n, K))
    U[np.arange(n), labels] = 1

    nc = np.sum(U, axis=0)  # (K,)
    P = (data @ U) / np.maximum(nc, 1e-10)  # (d, K)

    s_norm = np.sum(data * data)
    norm_P = np.sum(P * P, axis=0)  # (K,)

    t = 0
    v_obj_values[t] = s_norm - np.dot(nc, norm_P)

    while t < tmax - 1:
        t += 1
        D = -2 * (P.T @ data) + norm_P[:, np.newaxis]  # (K, n)
        last = labels.copy()
        labels = np.argmin(D, axis=0)

        if np.all(last == labels):
            break

        v_obj_values[t] = s_norm + np.sum(np.min(D, axis=0))

        U = np.zeros((n, K))
        U[np.arange(n), labels] = 1
        nc = np.sum(U, axis=0)
        P = (data @ U) / np.maximum(nc, 1e-10)
        norm_P = np.sum(P * P, axis=0)

    # Labels are returned 1-indexed
    return v_obj_values, labels + 1, P
