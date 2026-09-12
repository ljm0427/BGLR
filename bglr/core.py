"""
BGLR: Learning Manifold-Consistent Anchor Bases for Multi-View Short Text Clustering

Fits one configuration with explicit parameters: wires together anchor /
bipartite-graph construction (fast_multi_clr) with the ADMM solver (algo_qp),
then evaluates the learned representation. Parameter selection is not part of
this module; see run_grid_search.py for the label-free search.
"""

import time
import numpy as np
import torch

from .utils import normalize_fea
from .fast_multi_clr import fast_multi_clr
from .algo_qp import algo_qp
from .manifold import build_sparse_laplacian
from .metrics import clustering_measure, silhouette_score


class BGLR:
    """
    Anchor-based multi-view graph clustering.

    Runs a single configuration supplied by the caller.

    Parameters
    ----------
    n_clusters : int
    k2 : int, number of anchor-clusters
    m : int, anchors per anchor-cluster
    alpha, beta : float, regularization weights
    gamma : float, graph Laplacian weight
    k_nn, lap_knn : int, nearest neighbors for bipartite / Laplacian graphs
    max_iter, tol, device, seed, verbose
    aug_target, aug_index, num_views, adaptive_weight
    collect_state : bool, also expose converged internals through ``state_``
    """

    def __init__(
        self,
        n_clusters,
        k2,
        m,
        alpha=1.0,
        beta=1.0,
        gamma=0.0,
        k_nn=10,
        lap_knn=10,
        max_iter=200,
        tol=1e-6,
        device='cpu',
        seed=42,
        verbose=True,
        aug_target=2,
        aug_index=0,
        num_views=None,
        adaptive_weight=True,
        freeze_anchors=False,  # ablation: w/o learnable anchors
        save_snapshots=False,  # save Z_bar at iter 0,25,50,100,final for convergence viz
        collect_state=False,   # expose converged internals (Z, F, G) via state_
    ):
        self.freeze_anchors = freeze_anchors
        self.save_snapshots = save_snapshots
        self.collect_state = collect_state
        self.n_clusters = n_clusters
        self.k2 = k2
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k_nn = k_nn
        self.lap_knn = lap_knn
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.aug_target = aug_target
        self.aug_index = aug_index
        self.num_views = num_views
        self.adaptive_weight = adaptive_weight

        # Results
        self.representation_ = None  # fused representation Y (n, K)
        self.acc_ = None             # accuracy of the reported partition
        self.silhouette_ = None      # silhouette of that same partition
        self.params_ = None          # configuration used by this run
        self.snapshots_ = None       # convergence visualization data
        self.obj_vals_ = None        # objective value per ADMM iteration
        self.state_ = None           # converged internals, only when collect_state

    def fit(self, X, true_labels=None, aug_fea=None, times_clustering=10):
        """
        Fit one configuration with the parameters given to the constructor.

        Parameters
        ----------
        X : list of ndarray, X[iv]: (d_v, n)
        true_labels : ndarray (n,), optional
        aug_fea : list of ndarray, optional
        times_clustering : int, K-means repetitions for evaluation

        Returns
        -------
        self : BGLR
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if self.num_views is not None:
            X = X[:self.num_views]
        num_view = len(X)
        n = X[0].shape[1]

        if self.aug_target is not None and self.aug_target >= num_view:
            if self.verbose:
                print(f"Warning: aug_target={self.aug_target} >= num_views={num_view}. Disabling augmentation.")
            self.aug_target = None

        total_anchors = self.k2 * self.m
        if total_anchors > n:
            raise ValueError(f"total_anchors ({total_anchors}) > n ({n})")

        # Augmentation
        has_aug = aug_fea is not None and len(aug_fea) > 0
        if has_aug:
            aug_fea_norm = [normalize_fea(a, 0) for a in aug_fea]

        # Normalize base views
        X_base = [None] * num_view
        for iv in range(num_view):
            if has_aug and self.aug_target is not None and iv == self.aug_target:
                X_base[iv] = None
            else:
                X_base[iv] = normalize_fea(X[iv], 0)
        if has_aug and self.aug_target is not None:
            aug_idx = max(0, min(self.aug_index, len(aug_fea_norm) - 1))
            X_base[self.aug_target] = aug_fea_norm[aug_idx].copy()

        if self.verbose:
            print(f"\n  n={n}, clusters={self.n_clusters}, views={num_view}")
            print(f"  k2={self.k2}, m={self.m}, anchors={total_anchors}")
            print(f"  α={self.alpha}, β={self.beta}, γ={self.gamma}")

        # Laplacian (sparse, O(s n) storage)
        L_lap = None
        if self.gamma > 0 and num_view >= 1:
            if self.verbose:
                print(f"  Building sparse Laplacian (s={self.lap_knn})...")
            L_lap = build_sparse_laplacian(
                X_base[0], k_nn=self.lap_knn, device=self.device
            )
            if self.verbose:
                print(f"  {L_lap}")

        # Anchor selection + bipartite graph
        X2 = [X_base[iv].T for iv in range(num_view)]
        t0 = time.time()
        centers, B = fast_multi_clr(X2, self.k2, total_anchors, k=self.k_nn)
        U_in = [centers[iv].T for iv in range(num_view)]
        S_in = [B[iv].T for iv in range(num_view)]
        anchor_time = time.time() - t0

        # ADMM
        param = {
            'alpha': self.alpha, 'beta': self.beta,
            'gamma': self.gamma, 'L': L_lap, 'k2': self.k2,
            'maxiter': self.max_iter, 'tolfun': self.tol,
            'adaptive_weight': self.adaptive_weight,
            'freeze_anchors': self.freeze_anchors,
            'save_snapshots': self.save_snapshots,
            'collect_state': self.collect_state,
        }

        t1 = time.time()
        out = algo_qp(
            X_base, self.m, self.n_clusters, U_in, S_in, param, device=self.device
        )
        # Trailing optional elements: snapshots (save_snapshots), state (collect_state)
        Final_V1, Final_V2, obj_vals = out[0], out[1], out[2]
        tail = list(out[4:])
        snapshots = tail.pop(0) if (self.save_snapshots and tail) else None
        self.state_ = tail.pop(0) if tail else None
        admm_time = time.time() - t1

        if true_labels is not None:
            nmi2, acc2, pur2, _, _, pred_labels = clustering_measure(
                Final_V1, true_labels, times_clustering, self.seed,
                return_labels=True,
            )
            # Silhouette of the reported partition, computed on the fused
            # representation Y with the same K-means labels as ACC/NMI.
            sil = silhouette_score(Final_V1, pred_labels)
            self.acc_ = acc2
            self.silhouette_ = sil
            self.params_ = {'k2': self.k2, 'm': self.m,
                            'alpha': self.alpha, 'beta': self.beta,
                            'gamma': self.gamma}
            self.representation_ = Final_V1
            self.obj_vals_ = obj_vals
            if self.save_snapshots:
                self.snapshots_ = snapshots
            if self.verbose:
                print(f"  ACC:{acc2:.4f} NMI:{nmi2:.4f} Purity:{pur2:.4f} Silhouette:{sil:.4f}"
                      f" | anchor:{anchor_time:.1f}s admm:{admm_time:.1f}s")
        else:
            self.representation_ = Final_V1
            self.obj_vals_ = obj_vals
            if self.verbose:
                print(f"  iter={len(obj_vals)} obj={obj_vals[-1]:.4f}"
                      f" | anchor:{anchor_time:.1f}s admm:{admm_time:.1f}s")

        return self

    def get_representation(self):
        """Return the fused representation produced by the last ``fit``."""
        return self.representation_
