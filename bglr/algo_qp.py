"""
algo_qp: ADMM-based Multi-view Anchor Graph Optimization

Solves the joint optimization over:
  U{iv}: anchor matrices (d_v × k2*m)
  S{iv}: bipartite graph matrices (k2*m × n)
  T{iv}: cluster-to-anchor-cluster assignment (k2 × k)
  D{iv}: intermediate anchor structure (d_v × k2)
  R{iv}: ADMM auxiliary for the S simplex constraint (k2*m × n)
  Λ{iv}: ADMM dual for S (k2*m × n)
  J{iv}: ADMM auxiliary for T non-negativity (k2 × k)
  Γ{iv}: ADMM dual for T (k2 × k)
  Q: shared orthogonal clustering embedding (n × k)
  P{iv}: intermediate representation (k2*m × n)

Regularization terms: alpha on the anchor structure, beta on the cluster
consistency with Q, gamma on the TF-IDF similarity graph Laplacian.
"""

import numpy as np
import torch
from .utils import opt_s_torch


def _svd_fallback(M, full_matrices=False):
    """torch SVD with NaN-safe numpy fallback for ill-conditioned matrices."""
    try:
        return torch.linalg.svd(M, full_matrices=full_matrices)
    except torch._C._LinAlgError:
        # Clean NaN/Inf, then try numpy SVD
        M_np = M.detach().cpu().numpy()
        M_np = np.nan_to_num(M_np, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            U_np, s_np, Vt_np = np.linalg.svd(M_np, full_matrices=full_matrices)
        except np.linalg.LinAlgError:
            # Ultimate fallback: random orthogonal + zero singular values
            m, n = M_np.shape
            k = min(m, n) if full_matrices else min(m, n)
            U_np = np.eye(m, k)
            s_np = np.zeros(k)
            Vt_np = np.eye(k, n)
        dtype = M.dtype
        device = M.device
        return (torch.tensor(U_np, dtype=dtype, device=device),
                torch.tensor(s_np, dtype=dtype, device=device),
                torch.tensor(Vt_np, dtype=dtype, device=device))


def algo_qp(X, m, k, U, S, param, device='cpu'):
    """
    ADMM optimization for BGLR.

    Parameters
    ----------
    X : list of ndarray
        X[iv]: (d_v, n), each column is a sample for view iv.
    m : int
        Number of anchors per anchor-cluster.
    k : int
        Number of clusters.
    U : list of ndarray
        U[iv]: (d_v, k2*m), initial anchor matrices from FastmultiCLR.
    S : list of ndarray
        S[iv]: (k2*m, n), initial bipartite graph matrices from FastmultiCLR.
    param : dict with keys:
        alpha, beta : float, regularization weights
        gamma : float, graph Laplacian regularization weight (0=disabled)
        L : bglr.manifold.SparseLaplacian, pre-built sparse graph Laplacian
            (optional). It is only ever used through ``apply_right`` and
            ``quadratic_form``, so the dense n x n form is never materialized.
        k2 : int, number of anchor-clusters
        maxiter : int, max ADMM iterations (default 200)
        tolfun : float, reserved for a configurable tolerance. It is read but
            not used: the convergence check below applies fixed thresholds.
        adaptive_weight : bool, inverse-residual view weights (default True)
        freeze_anchors : bool, ablation: keep anchors at their initialization
        save_snapshots : bool, also record Z_bar at iterations 0/25/50/100/final
        collect_state : bool, when True the converged internals
            (Z, F, G, view_weights) are appended as a final ``state`` dict.
            Needed by post-optimization fusion routines that live outside this
            solver (default False, keeps the default return signature).
    device : str, 'cpu' or 'cuda'

    Returns
    -------
    Final_V1 : ndarray (n, k)
        Fused representation Y = Σ_v w_v Z_v^T F H_v, produced by the
        post-optimization residual calibration.
    Final_V2 : ndarray (n, k)
        Final shared orthogonal clustering embedding.
    obj : list
        Objective function values per iteration.
    residual : list
        Primal split residual r_t = max_v ||R_v - Z_v||_inf per iteration,
        i.e. the constraint violation of the R_v = Z_v splitting.
    snapshots : dict, optional
        Appended only when ``param['save_snapshots']`` is True.
    state : dict, optional
        Appended only when ``param['collect_state']`` is True.
    """
    num_view = len(X)
    n = X[0].shape[1]

    alpha = param['alpha']
    beta = param['beta']
    gamma_lap = param.get('gamma', 0.0)
    L_lap = param.get('L', None)  # pre-built graph Laplacian
    adaptive_weight = param.get('adaptive_weight', True)
    freeze_anchors = param.get('freeze_anchors', False)  # ablation: w/o learnable anchors
    save_snapshots = param.get('save_snapshots', False)
    collect_state = param.get('collect_state', False)
    k2 = param['k2']
    maxiter = param.get('maxiter', 200)
    # Reserved for a future configurable tolerance: the convergence check below
    # currently uses hard-coded thresholds (see "Convergence check").
    tolfun = param.get('tolfun', 1e-6)

    # ---- Choose dtype based on device ----
    # Always use float32 for GPU; float64 only for CPU
    dtype = torch.float32 if device == 'cuda' else torch.float64

    # ---- Convert to torch tensors ----
    X_t = [torch.tensor(X[iv], dtype=dtype, device=device) for iv in range(num_view)]
    U_t = [torch.tensor(U[iv], dtype=dtype, device=device) for iv in range(num_view)]
    S_t = [torch.tensor(S[iv], dtype=dtype, device=device) for iv in range(num_view)]

    # ---- Initialize ADMM variables ----
    # W: block-diagonal structure matrix (k2*m × k2), W'W = I
    W = torch.zeros(k2 * m, k2, dtype=dtype, device=device)
    for i in range(k2):
        start = i * m
        end = (i + 1) * m
        W[start:end, i] = 1.0
    # Normalize columns of W
    W_norm = torch.sqrt(torch.sum(W ** 2, dim=0, keepdims=True))
    W = W / torch.clamp(W_norm, min=1e-14)

    # T{iv}: cluster-to-anchor-cluster assignment (k2 × k)
    # Initialize with block structure
    T = []
    num_elements = k2 // k  # number of anchor-clusters per cluster
    for iv in range(num_view):
        T_iv = torch.zeros(k2, k, dtype=dtype, device=device)
        for i in range(k):
            start = i * num_elements
            end = (i + 1) * num_elements
            T_iv[start:end, i] = 1.0
        # Column normalize
        T_norm = torch.sqrt(torch.sum(T_iv ** 2, dim=0, keepdims=True))
        T_iv = T_iv / torch.clamp(T_norm, min=1e-14)
        T.append(T_iv)

    # D{iv} = U{iv} * W  (d_v × k2)
    D = [U_t[iv] @ W for iv in range(num_view)]

    # Initialize Q: T{iv}' * W' * S{iv} → (k × n)
    # svd returns (U, S, Vh) with Vh = V', so V = Vh.T
    Q_init = torch.zeros(k, n, dtype=dtype, device=device)
    for iv in range(num_view):
        Q_init = Q_init + T[iv].T @ (W.T @ S_t[iv])  # (k × k2) * (k2 × k2m) * (k2m × n) = (k × n)
    Unew, _, Vh = _svd_fallback(Q_init)
    Q = Vh.T @ Unew.T  # (n, k)

    # Initialize other variables
    J = [torch.zeros(k2, k, dtype=dtype, device=device) for _ in range(num_view)]
    Gamma = [torch.zeros(k2, k, dtype=dtype, device=device) for _ in range(num_view)]
    R = [torch.zeros(k2 * m, n, dtype=dtype, device=device) for _ in range(num_view)]
    Lambda = [torch.zeros(k2 * m, n, dtype=dtype, device=device) for _ in range(num_view)]

    # P{iv} = S{iv} * Q  (k2*m × n) * (n × k) = (k2*m × k)
    P = [S_t[iv] @ Q for iv in range(num_view)]

    # ADMM penalty parameter
    rho_admm = 1e-7
    rho_scale = 1.5

    obj = []
    residual_list = []

    # Convergence visualization snapshots (iter 0, 25, 50, 100, final)
    snapshot_iters = {25, 50, 100}
    snapshots = {}
    if save_snapshots:
        with torch.no_grad():
            Vm0 = torch.zeros(n, k2 * m, dtype=dtype, device=device)
            for iv in range(num_view):
                Vm0 = Vm0 + S_t[iv].T
            Vm0 = Vm0 / num_view
            snapshots[0] = Vm0.cpu().numpy()

    iter_count = 1
    Isconverg = False

    I_k2m = torch.eye(k2 * m, dtype=dtype, device=device)

    # ---- Main ADMM Loop ----
    while not Isconverg:

        # ========== Step 1: Update U{iv} (each view independently) ==========
        if not freeze_anchors:
            for iv in range(num_view):
                U_t[iv] = torch.linalg.solve(
                    S_t[iv] @ S_t[iv].T + alpha * I_k2m,
                    (X_t[iv] @ S_t[iv].T + alpha * D[iv] @ W.T).T
                ).T

        # ========== Steps 2-7: Update S, R, T, J, Lambda, Gamma, D (each view) ==========
        mu2 = rho_admm / 2.0
        tmpQ = torch.zeros(k, n, dtype=dtype, device=device)

        for iv in range(num_view):
            # ---- Step 2: Update S{iv} ----
            # LHS: U^T U + (rho/2 + beta)·I
            lhs_coeff = mu2 + beta
            lhs = U_t[iv].T @ U_t[iv] + lhs_coeff * I_k2m
            rhs = (U_t[iv].T @ X_t[iv]
                   + beta * P[iv] @ Q.T
                   + mu2 * R[iv]
                   + Lambda[iv] / 2.0)

            # Graph Laplacian regularization: add -γ * S * L to rhs
            # Sparse edge multiplication: O(M * nnz(L)) time, O(M n) memory.
            # Clamp Laplacian term to prevent NaN explosion when γ is large
            if gamma_lap > 0 and L_lap is not None:
                lap_term = gamma_lap * L_lap.apply_right(S_t[iv])
                lap_term = torch.clamp(lap_term, -1e6, 1e6)
                rhs = rhs - lap_term

            S_t[iv] = torch.linalg.solve(lhs, rhs)
            # Clamp S to prevent NaN propagation
            S_t[iv] = torch.clamp(S_t[iv], -1e6, 1e6)

            # ---- Step 3: Update R{iv} — simplex projection (GPU-native) ----
            R[iv] = opt_s_torch(S_t[iv] - Lambda[iv] / rho_admm, 1.0)

            # ---- Step 4: Update T{iv} — orthogonal Procrustes ----
            tmp = rho_admm * J[iv] + Gamma[iv]  # (k2, k)

            Unew2, _, Vnew2 = _svd_fallback(tmp)
            T[iv] = Unew2 @ Vnew2

            # Accumulate for Q update
            tmpQ = tmpQ + beta * P[iv].T @ S_t[iv]

            # ---- Step 5: Update D{iv} ----
            D[iv] = U_t[iv] @ W

            # ---- Step 6: Update J{iv} — non-negativity projection ----
            J[iv] = torch.clamp(T[iv] - Gamma[iv] / rho_admm, min=0)

            # ---- Step 7: Update ADMM dual variables ----
            Lambda[iv] = Lambda[iv] + rho_admm * (R[iv] - S_t[iv])
            Gamma[iv] = Gamma[iv] + rho_admm * (J[iv] - T[iv])

        # ========== Step 8: Update Q — shared cluster assignment ==========
        Unew3, _, Vh3 = _svd_fallback(tmpQ)
        Q = Vh3.T @ Unew3.T  # (n, k)

        # ========== Step 9: Update P{iv} (pure GPU) ==========
        for iv in range(num_view):
            P[iv] = S_t[iv] @ Q

        # ========== Update ADMM penalty ==========
        rho_admm = min(rho_scale * rho_admm, 1e10)

        # ========== Compute objective and split residual (minimal GPU syncs) ==========
        # Only sync when needed for convergence check
        dnorm = 0.0
        primal_res = 0.0

        for iv in range(num_view):
            err1 = X_t[iv] - U_t[iv] @ S_t[iv]
            err3 = U_t[iv].T - W @ D[iv].T
            err4 = S_t[iv] - P[iv] @ Q.T

            dnorm += (torch.sum(err1 ** 2).item()
                      + alpha * torch.sum(err3 ** 2).item()
                      + beta * torch.sum(err4 ** 2).item())

            # Graph Laplacian term: γ * tr(S L S^T), edge form (O(M nnz))
            if gamma_lap > 0 and L_lap is not None:
                dnorm += gamma_lap * L_lap.quadratic_form(S_t[iv]).item()

            # Primal residual of the R = Z split: r_t = max_v ||R_v - Z_v||_inf
            primal_res = max(primal_res,
                             torch.max(torch.abs(R[iv] - S_t[iv])).item())

        obj.append(dnorm)
        residual_list.append(primal_res)

        # Save snapshot at checkpoint iterations (only 3 times total)
        if save_snapshots and iter_count in snapshot_iters:
            with torch.no_grad():
                Vm = torch.zeros(n, k2 * m, dtype=dtype, device=device)
                for iv in range(num_view):
                    Vm = Vm + S_t[iv].T
                Vm = Vm / num_view
                snapshots[iter_count] = Vm.cpu().numpy()

        # Convergence check: stop only when the relative objective change and the
        # change of the split residual are both small.
        if iter_count >= 2:
            d_res = abs(residual_list[-2] - residual_list[-1])
            d_obj = abs(obj[-2] - obj[-1]) / max(abs(obj[-2]), 1e-10)

            if d_res < 1e-7 and d_obj < 1e-4:
                Isconverg = True

        if iter_count > maxiter:
            Isconverg = True

        iter_count += 1

    # Save final snapshot (converged state)
    if save_snapshots:
        with torch.no_grad():
            Vm = torch.zeros(n, k2 * m, dtype=dtype, device=device)
            for iv in range(num_view):
                Vm = Vm + S_t[iv].T
            Vm = Vm / num_view
            snapshots['final'] = Vm.cpu().numpy()

    # ========== Final Output ==========
    # Post-optimization residual calibration and fusion.
    # The weights live outside the ADMM loop: G is learned as an equal-contribution
    # consensus structure, and the calibration only re-weights the final decision.
    with torch.no_grad():
        H_list, residuals = [], []
        for iv in range(num_view):
            H_tilde = W.T @ S_t[iv] @ Q                    # F^T Z_v G  (k2, k)
            U_h, _, Vh_h = _svd_fallback(H_tilde)
            H_v = U_h @ Vh_h                               # Procrustes (k2, k)
            H_list.append(H_v)
            err = S_t[iv] - W @ H_v @ Q.T
            residuals.append(torch.sum(err ** 2).item())   # view residual e_v

        if adaptive_weight:
            res = torch.tensor(residuals, dtype=dtype, device=device)
            inv = 1.0 / (res + 1e-12)                      # w_v ∝ 1/(e_v + ε)
            view_weights = inv / inv.sum()
        else:
            # Equal weights for the ablation w/o residual calibration
            view_weights = torch.ones(num_view, dtype=dtype, device=device) / num_view

        # Final_V1: fused representation Y = Σ_v w_v Z_v^T F H_v → (n, k)
        Y = torch.zeros(n, k, dtype=dtype, device=device)
        for iv in range(num_view):
            Y = Y + view_weights[iv] * (W.T @ S_t[iv]).T @ H_list[iv]

    Final_V1 = Y.cpu().numpy()

    # Final_V2: shared orthogonal clustering embedding G → (n, k)
    Final_V2 = Q.cpu().numpy()

    # Optional converged internals, requested by external post-optimization
    # fusion / model-selection routines.
    state = None
    if collect_state:
        state = {
            'Z': [S_t[iv].detach().cpu().numpy() for iv in range(num_view)],
            'F': W.detach().cpu().numpy(),
            'G': Q.detach().cpu().numpy(),
            'view_weights': view_weights.detach().cpu().numpy(),
        }

    out = [Final_V1, Final_V2, obj, residual_list]
    if save_snapshots:
        out.append(snapshots)
    if state is not None:
        out.append(state)
    return tuple(out)
