"""
Sparse lexical-manifold graph Laplacian.

The Laplacian is built from a TF-IDF-SVD s-nearest-neighbor cosine graph:

    W_hat_ij = max(cos(x_i, x_j), 0)   for j in N_s(i) U {i},  0 otherwise
    W        = (W_hat + W_hat^T) / 2
    D        = Diag(W 1_n)
    L        = I_n - D^{-1/2} W D^{-1/2}

Only the retained edges are ever materialized, so the graph costs O(s n)
storage. The dense n x n similarity matrix, the dense W and the dense L are
never formed; the k-nearest-neighbor search is chunked so that its peak dense
block is O(chunk * n) rather than O(n^2). Exact neighbor search remains part
of preprocessing.

The solver only needs two operations, both provided in edge form:

    Z L         -> SparseLaplacian.apply_right(Z)
    tr(Z L Z^T) -> SparseLaplacian.quadratic_form(Z)

Each costs O(M * nnz(L)) time and O(M n) memory, instead of the O(M n^2) time
and O(n^2) memory of a dense Laplacian.
"""

from __future__ import annotations

import torch


class SparseLaplacian:
    """
    Symmetrically normalized graph Laplacian stored in sparse edge form.

    Attributes
    ----------
    w_sparse : torch.Tensor (n, n) sparse COO
        Coalesced symmetrized affinity matrix W.
    degree_inv_sqrt : torch.Tensor (n,)
        D^{-1/2}, used by both operations below.
    n : int
        Number of samples.
    nnz : int
        Number of stored edges, O(s n).
    """

    def __init__(self, w_sparse, degree_inv_sqrt, n, dtype=None, device=None):
        self.w_sparse = w_sparse.coalesce()
        self.degree_inv_sqrt = degree_inv_sqrt
        self.n = int(n)
        self.nnz = int(self.w_sparse.indices().shape[1])
        self.dtype = dtype if dtype is not None else self.w_sparse.dtype
        self.device = device if device is not None else self.w_sparse.device

    # -- edge accessors ----------------------------------------------------
    @property
    def edges(self):
        """(2, nnz) int64 indices of the retained affinities."""
        return self.w_sparse.indices()

    @property
    def values(self):
        """(nnz,) symmetrized affinity values W_ij."""
        return self.w_sparse.values()

    # -- operations used by the solver -------------------------------------
    def _right_multiply(self, Y):
        """Return Y @ W for a dense (M, n) tensor.

        W is symmetric, so ``W @ Y^T`` (a supported sparse-dense product) is
        just ``(Y @ W)^T``. Cost O(M * nnz), memory O(M n).
        """
        return torch.sparse.mm(self.w_sparse, Y.t()).t()

    def apply_right(self, Z):
        """Return Z @ L for a dense (M, n) tensor, without forming L.

        Z L = Z - (Z D^{-1/2}) W D^{-1/2}
        """
        d = self.degree_inv_sqrt.unsqueeze(0)          # (1, n)
        Y = Z * d                                      # Z D^{-1/2}
        return Z - self._right_multiply(Y) * d

    def quadratic_form(self, Z):
        """Return tr(Z L Z^T) for a dense (M, n) tensor.

        tr(Z L Z^T) = ||Z||_F^2 - tr(Y W Y^T),  Y = Z D^{-1/2}
        and the second trace is the entrywise product <Y W, Y>, which avoids
        the n x n matrix Y^T Y. Cost O(M * nnz), memory O(M n).
        """
        d = self.degree_inv_sqrt.unsqueeze(0)
        Y = Z * d
        return torch.sum(Z ** 2) - torch.sum(self._right_multiply(Y) * Y)

    # -- debug helper ------------------------------------------------------
    def to_dense(self):
        """Materialize the dense L. Debug/visualisation only: this is exactly
        the O(n^2) object the sparse representation exists to avoid."""
        d = self.degree_inv_sqrt
        return (torch.eye(self.n, dtype=self.dtype, device=self.device)
                - self.w_sparse.to_dense() * d.unsqueeze(0) * d.unsqueeze(1))

    def __repr__(self):
        return (f"SparseLaplacian(n={self.n}, nnz={self.nnz}, "
                f"dtype={self.dtype}, device={self.device})")


def build_sparse_laplacian(X, k_nn=10, device='cpu', dtype=None,
                           chunk_size=None, memory_budget_mb=128):
    """
    Build the sparse symmetrically normalized graph Laplacian.

    Parameters
    ----------
    X : ndarray or torch.Tensor, shape (d, n)
        Lexical (TF-IDF + SVD) features; each column is one short text.
    k_nn : int
        Number of nearest neighbors s of the retained graph (default 10).
    device : str, 'cpu' or 'cuda'.
    dtype : torch.dtype, optional
        Defaults to float32 on CUDA and float64 on CPU, matching the solver.
    chunk_size : int, optional
        Rows of the similarity matrix computed at once. If None it is derived
        from ``memory_budget_mb`` so that one block stays within the budget.
    memory_budget_mb : int
        Approximate peak-memory budget for one similarity block (default 128 MB).

    Returns
    -------
    SparseLaplacian
        Storage is O(s n); no n x n dense matrix is ever allocated.
    """
    if dtype is None:
        dtype = torch.float32 if device == 'cuda' else torch.float64

    X_t = torch.as_tensor(X, dtype=dtype, device=device)
    if X_t.dim() != 2:
        raise ValueError(f"X must be 2-D (d, n), got shape {tuple(X_t.shape)}")
    _, n = X_t.shape
    s = int(min(k_nn + 1, n))          # s neighbors + self

    # Column-normalized features: cosine similarity becomes an inner product.
    Xn = X_t / torch.norm(X_t, dim=0, keepdim=True).clamp(min=1e-12)

    if chunk_size is None:
        itemsize = 8 if dtype == torch.float64 else 4
        chunk_size = max(1, min(n, int(memory_budget_mb) * 1024 * 1024
                                // (max(n, 1) * itemsize)))

    # ---- Chunked exact top-k search: peak dense block is O(chunk * n) ----
    col_parts, val_parts = [], []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = Xn[:, start:end].t() @ Xn          # (b, n) cosine similarities
        block = block.clamp(min=0)                 # keep the nonnegative part
        topv, topi = torch.topk(block, k=s, dim=1)  # (b, s) values / indices
        col_parts.append(topi)
        val_parts.append(topv)

    rows = torch.arange(n, device=device).unsqueeze(1).expand(n, s)
    cols = torch.cat(col_parts, dim=0)
    vals = torch.cat(val_parts, dim=0)

    # The self term is kept (j in N_s(i) U {i}); if ties pushed the diagonal
    # out of the top-s, add it back with its exact value of 1.
    has_self = (cols == rows).any(dim=1)           # (n,)
    rows = rows.reshape(-1)
    cols = cols.reshape(-1)
    vals = vals.reshape(-1)
    missing = torch.nonzero(~has_self, as_tuple=False).squeeze(1)
    if missing.numel() > 0:
        rows = torch.cat([rows, missing])
        cols = torch.cat([cols, missing])
        vals = torch.cat([vals, torch.ones_like(missing, dtype=dtype)])

    # ---- Symmetrize W = (W_hat + W_hat^T) / 2 in edge form ----
    idx = torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])], dim=0)
    w_vals = torch.cat([vals, vals]) / 2.0
    W = torch.sparse_coo_tensor(idx, w_vals, (n, n),
                                dtype=dtype, device=device).coalesce()

    # ---- D and L = I - D^{-1/2} W D^{-1/2} ----
    degree = torch.sparse.sum(W, dim=1).to_dense()          # (n,), O(nnz)
    degree_inv_sqrt = 1.0 / torch.sqrt(degree.clamp(min=1e-12))

    return SparseLaplacian(W, degree_inv_sqrt, n, dtype=dtype, device=device)
