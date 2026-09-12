#!/usr/bin/env python
"""
run_grid_search.py — two-stage label-free parameter selection for BGLR.

Protocol
--------
Stage 1 — anchor structure
    Fixed coefficients: alpha_s = tau_A = alpha = 1, beta = 1, gamma = 1.
    Searched jointly: the anchor-group ratio K_a/K and the group size m, with
        K  <= 30 :  K_a/K in {10, 8, 4},  m in {2, 8, 16}
        K  >  30 :  K_a/K in {1, 2, 4},   m in {2, 3, 4}
    The selected pair fixes the total number of anchors M = K_a * m.
    Large-K datasets already have K_a ~ K, so their m range is kept small to
    avoid an unnecessarily large M; use --ka-ratios / --m-values to override
    either range.

Stage 2 — trade-off coefficients
    K_a, m (hence M) are frozen from stage 1 and the complete 4 x 4 x 4 grid is
    evaluated with alpha_s = tau_A = alpha tied:
        alpha in {0.01, 0.1, 1, 10}
        beta  in {0.1, 1, 10, 100}
        gamma in {0.01, 0.1, 1, 10}

Selection score (identical in both stages)
    Each candidate is optimized by ADMM and mapped to the fused representation
    Y. Ten K-means repetitions are then applied to Y, and the exact Euclidean
    silhouette coefficient is averaged over the ten runs. The
    candidate with the largest mean silhouette is retained.

Label discipline
    Ground-truth labels are never touched during selection: ``fit`` is called
    with ``true_labels=None`` and no ACC/NMI is computed. The dataset cluster
    count K is used only to choose the anchor-group ratio range, because the
    formulation assumes K is known. Any labelled evaluation is opt-in through
    ``--eval-final`` and is reported strictly after selection has finished.

Usage
-----
    # full two-stage search on one dataset
    python run_grid_search.py --dataset AGNews --device cuda

    # stage 1 only (anchor structure)
    python run_grid_search.py --dataset AGNews --stage 1

    # stage 2 only, reusing the stage-1 result stored in the checkpoint
    python run_grid_search.py --dataset AGNews --stage 2

    # evaluate one frozen configuration, no search
    python run_grid_search.py --dataset AGNews --stage fixed \
        --ka 40 --m 16 --alpha 0.01 --beta 100 --gamma 1

    # print the candidate ranges without running anything
    python run_grid_search.py --dataset GoogleNews-TS --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sk_silhouette

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# NOTE: BGLR (torch) and the dataset loader are imported inside main() so that
# --dry-run and the grid/fusion helpers can be used without the full stack.


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
SELECTION_SEED = 42                 # seed used for the selection run
KMEANS_REPS = 10                    # K-means repetitions per candidate
SMALL_K_THRESHOLD = 30              # K <= 30 -> "small", K > 30 -> "large"
KA_RATIOS_SMALL_K = (10, 8, 4)      # K_a / K for K <= 30
KA_RATIOS_LARGE_K = (1, 2, 4)       # K_a / K for K > 30
# Anchors per anchor group, also K-dependent: a large-K dataset already has
# K_a ~ K, so a large m would inflate M = K_a * m without adding structure.
# In practice the selected m stays at the small end (2) for large-K datasets.
M_VALUES_SMALL_K = (2, 8, 16)       # m for K <= 30
M_VALUES_LARGE_K = (2, 3, 4)        # m for K > 30
ALPHA_GAMMA_GRID = (0.01, 0.1, 1.0, 10.0)
BETA_GRID = (0.1, 1.0, 10.0, 100.0)
STAGE1_ALPHA = 1.0                  # alpha_s = tau_A = alpha, tied
STAGE1_BETA = 1.0
STAGE1_GAMMA = 1.0
EPS = 1e-12

# Fixed solver / graph settings shared by every candidate
K_NN = 10
LAP_KNN = 10
MAX_ITER = 200
TOL = 1e-6


# ---------------------------------------------------------------------------
# Candidate grids
# ---------------------------------------------------------------------------
def ka_ratios_for(K):
    """Anchor-group ratio range. Different range for small and large K."""
    return KA_RATIOS_SMALL_K if K <= SMALL_K_THRESHOLD else KA_RATIOS_LARGE_K


def m_values_for(K):
    """Anchors-per-group range. Different range for small and large K."""
    return M_VALUES_SMALL_K if K <= SMALL_K_THRESHOLD else M_VALUES_LARGE_K


def parse_ka_ratios(text):
    """'1' or '1,2' -> tuple of ratios; floats keep values such as 0.5 exact."""
    return tuple(float(tok) for tok in text.split(",") if tok.strip())


def parse_int_list(text):
    """'2,3,4' -> (2, 3, 4)."""
    return tuple(int(tok) for tok in text.split(",") if tok.strip())


def stage1_candidates(K, ratios=None, m_values=None):
    """[(K_a/K, m), ...] — anchor structure only."""
    ratios = ka_ratios_for(K) if ratios is None else tuple(ratios)
    m_values = m_values_for(K) if m_values is None else tuple(m_values)
    return [(ratio, m) for ratio, m in product(ratios, m_values)]


def stage2_candidates():
    """[(alpha, beta, gamma), ...] — 4 x 4 x 4 trade-off grid."""
    return [(a, b, g) for a, b, g in product(ALPHA_GAMMA_GRID, BETA_GRID, ALPHA_GAMMA_GRID)]


def cfg_key(cfg):
    return "ka{ka}_m{m}_a{alpha:g}_b{beta:g}_g{gamma:g}".format(**cfg)


def rec_key(stage, cfg):
    """Checkpoint key: the stage is part of the key.

    Stage 1 fixes alpha = beta = gamma = 1, and the stage-2 grid also contains
    that point. Keying by stage as well keeps the two stages independent, so a
    cached stage-1 record can never shadow a stage-2 candidate.
    """
    return f"s{stage}:{cfg_key(cfg)}"


# ---------------------------------------------------------------------------
# Post-optimization fusion
# ---------------------------------------------------------------------------
def procrustes(H_tilde):
    """Orthogonal Procrustes solution H = U V^T of the thin SVD H_tilde = U S V^T."""
    U, _, Vt = np.linalg.svd(H_tilde, full_matrices=False)
    return U @ Vt


def fused_representation(Z_list, F, G, eps=EPS):
    """
    Residual-based view calibration and fusion.

    Parameters
    ----------
    Z_list : list of ndarray, Z[v] is (M, n) — sample-to-anchor coordinates
    F : ndarray (M, K_a) — fixed normalized anchor-group indicator
    G : ndarray (n, K)   — shared orthogonal clustering embedding

    Returns
    -------
    Y : ndarray (n, K) — fused representation
    weights : ndarray (V,) — calibration weights, sum to one
    residuals : ndarray (V,) — view reconstruction residuals e_v
    """
    residuals, H_list = [], []
    for Z in Z_list:
        H_v = procrustes(F.T @ Z @ G)
        H_list.append(H_v)
        residuals.append(float(np.sum((Z - F @ H_v @ G.T) ** 2)))

    residuals = np.asarray(residuals, dtype=np.float64)
    inv = 1.0 / (residuals + eps)
    weights = inv / inv.sum()

    Y = np.zeros((Z_list[0].shape[1], G.shape[1]), dtype=np.float64)
    for Z, H_v, w_v in zip(Z_list, H_list, weights):
        Y += w_v * (F.T @ Z).T @ H_v
    return Y, weights, residuals


def selection_score(Y, n_clusters, reps=KMEANS_REPS, seed=SELECTION_SEED, normalize=False):
    """
    Mean exact Euclidean silhouette over ``reps`` K-means repetitions on Y.

    Only Y is used — no ground-truth labels are involved.
    """
    data = Y
    if normalize:
        data = Y / np.clip(np.linalg.norm(Y, axis=1, keepdims=True), 1e-14, None)

    per_rep = []
    for r in range(reps):
        km = KMeans(n_clusters=n_clusters, n_init=1, init="k-means++",
                    random_state=seed + r)
        labels = km.fit_predict(data)
        if len(np.unique(labels)) < 2:
            per_rep.append(0.0)
            continue
        per_rep.append(float(sk_silhouette(data, labels, metric="euclidean")))
    return float(np.mean(per_rep)), per_rep


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------
def evaluate_candidate(X, aug_fea, n_clusters, cfg, args):
    """Run one candidate label-free and return its selection record."""
    from bglr import BGLR     # lazy: keeps --dry-run free of the torch dependency

    n = X[0].shape[1]
    M = cfg["ka"] * cfg["m"]
    if M > n:
        raise ValueError(f"infeasible anchor count M={M} > n={n}")

    model = BGLR(
        n_clusters=n_clusters,
        k2=cfg["ka"],
        m=cfg["m"],
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        gamma=cfg["gamma"],
        k_nn=K_NN,
        lap_knn=LAP_KNN,
        max_iter=MAX_ITER,
        tol=TOL,
        device=args.device,
        seed=args.select_seed,
        verbose=False,
        aug_target=2,
        aug_index=0,
        num_views=None,
        adaptive_weight=True,
        collect_state=True,
    )

    t0 = time.time()
    model.fit(X, true_labels=None, aug_fea=aug_fea, times_clustering=1)
    fit_seconds = time.time() - t0

    state = model.state_
    Y, weights, residuals = fused_representation(
        state["Z"], state["F"], state["G"]
    )
    score, per_rep = selection_score(
        Y, n_clusters, reps=args.reps, seed=args.select_seed,
        normalize=args.normalize_y,
    )

    return {
        "key": cfg_key(cfg),
        "params": dict(cfg),
        "M": M,
        "ok": True,
        "error": None,
        "score": score,
        "per_rep": per_rep,
        "view_weights": [float(w) for w in weights],
        "residuals": [float(r) for r in residuals],
        "iterations": len(model.obj_vals_) if model.obj_vals_ else None,
        "fit_seconds": fit_seconds,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def load_checkpoint(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path, records):
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp, path)


def best_from_records(records, stage):
    ok = [r for r in records if r.get("ok") and r.get("stage") == stage]
    if not ok:
        return None
    return max(ok, key=lambda r: r["score"])


# ---------------------------------------------------------------------------
# Running one stage
# ---------------------------------------------------------------------------
def run_candidates(stage_name, candidates, X, aug_fea, n_clusters, args,
                   records, done, checkpoint_path):
    """Evaluate every candidate that is not already in the checkpoint."""
    total = len(candidates)
    for idx, cfg in enumerate(candidates, start=1):
        key = rec_key(stage_name, cfg)
        if key in done:
            print(f"  [{idx:>3}/{total}] {cfg_key(cfg)}  -> cached")
            continue

        print(f"  [{idx:>3}/{total}] {key}  ...", end="", flush=True)
        t0 = time.time()
        try:
            rec = evaluate_candidate(X, aug_fea, n_clusters, cfg, args)
            status = (f" SC={rec['score']:.4f}  iters={rec['iterations']}"
                      f"  w=[{'/'.join(f'{w:.3f}' for w in rec['view_weights'])}]"
                      f"  {time.time() - t0:.1f}s")
        except Exception as exc:                                   # noqa: BLE001
            rec = {
                "key": key, "params": dict(cfg), "M": cfg["ka"] * cfg["m"],
                "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "score": None,
            }
            status = f" SKIPPED ({rec['error']})"

        rec["stage"] = stage_name
        rec["key"] = key
        records.append(rec)
        if rec["ok"]:
            done.add(key)
        save_checkpoint(checkpoint_path, records)
        print(status)

    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_stage_summary(records, stage, title, top=5, keys=None):
    """Rank the results of one stage.

    ``keys`` restricts the ranking to the candidate set of the current run, so a
    restricted grid is never out-ranked by cached records from a wider grid.
    """
    ok = sorted([r for r in records
                 if r.get("ok") and r.get("stage") == stage
                 and (keys is None or r.get("key") in keys)],
                key=lambda r: r["score"], reverse=True)
    skipped = [r for r in records if r.get("stage") == stage and not r.get("ok")]

    print(f"\n  {title}: {len(ok)} evaluated, {len(skipped)} skipped")
    if not ok:
        return None
    for r in ok[:top]:
        p = r["params"]
        print(f"    SC={r['score']:.4f}  Ka={p['ka']}  m={p['m']}  M={r['M']}"
              f"  alpha={p['alpha']:g}  beta={p['beta']:g}  gamma={p['gamma']:g}")
    return ok[0]


def print_final(best1, best2, n_clusters):
    if best2 is None and best1 is None:
        print("\nNo candidate could be evaluated.")
        return
    chosen = best2 if best2 is not None else best1
    p = chosen["params"]

    print("\n" + "=" * 78)
    print("Selected configuration (label-free, selection seed 42)")
    print("=" * 78)
    print(f"  dataset clusters K      : {n_clusters}")
    # Derive the ratio from the chosen configuration itself: when stage 2 is run
    # with an explicit --ka, the stage-1 ranking of this invocation is empty and
    # a cached record from a wider grid must not be used here.
    print(f"  K_a/K                   : {p['ka'] / n_clusters:g}")
    print(f"  K_a (anchor groups)     : {p['ka']}")
    print(f"  m (anchors per group)   : {p['m']}")
    print(f"  M (total anchors)       : {p['ka'] * p['m']}")
    print(f"  alpha_s = tau_A = alpha : {p['alpha']:g}")
    print(f"  beta                    : {p['beta']:g}")
    print(f"  gamma                   : {p['gamma']:g}")
    print(f"  SC (mean silhouette)    : {chosen['score']:.4f}")
    print("=" * 78)
    print("\nRe-run this configuration with:")
    print(f"  python run_clustering.py --mode cluster --dataset <name> "
          f"--k2 {p['ka']} --m {p['m']} --alpha {p['alpha']:g} "
          f"--beta {p['beta']:g} --gamma {p['gamma']:g}")


def evaluate_frozen(X, true_labels, aug_fea, n_clusters, cfg, args):
    """Labelled evaluation of the frozen configuration (NOT used for selection)."""
    from bglr import BGLR, clustering_measure

    seeds = [int(s) for s in args.eval_seeds.split(",") if s.strip()]
    print(f"\n[post-selection evaluation] {'/'.join(str(s) for s in seeds)}")
    accs, nmis = [], []
    for seed in seeds:
        model = BGLR(
            n_clusters=n_clusters, k2=cfg["ka"], m=cfg["m"],
            alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"],
            k_nn=K_NN, lap_knn=LAP_KNN, max_iter=MAX_ITER, tol=TOL,
            device=args.device, seed=seed, verbose=False,
            aug_target=2, aug_index=0, num_views=None, adaptive_weight=True,
        )
        model.fit(X, true_labels=true_labels, aug_fea=aug_fea,
                  times_clustering=args.times_clustering)
        nmi, acc, _, _, _ = clustering_measure(
            model.representation_, true_labels, args.times_clustering, seed
        )
        accs.append(acc)
        nmis.append(nmi)
        print(f"    seed {seed:>3}  ACC={acc:.4f}  NMI={nmi:.4f}")

    accs, nmis = np.asarray(accs), np.asarray(nmis)
    print(f"    mean over {len(seeds)} seeds  "
          f"ACC={accs.mean() * 100:.2f}+-{accs.std(ddof=1) * 100:.2f}  "
          f"NMI={nmis.mean() * 100:.2f}+-{nmis.std(ddof=1) * 100:.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage label-free grid search for BGLR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="tweet",
                        help="Dataset name (datasets/<name>.mat)")
    parser.add_argument("--data_dir", type=str, default="datasets",
                        help="Directory holding the .mat datasets")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "1", "2", "fixed"],
                        help="Which stage(s) to run")

    # Search knobs
    parser.add_argument("--reps", type=int, default=KMEANS_REPS,
                        help="K-means repetitions per candidate (default 10)")
    parser.add_argument("--select-seed", type=int, default=SELECTION_SEED,
                        help="Selection seed (default 42)")
    parser.add_argument("--normalize-y",
                        type=lambda x: x.lower() != 'false', default=True,
                        help="L2-normalize the rows of Y before K-means / silhouette. "
                             "Default True so that selection matches the evaluation "
                             "path of bglr.metrics.kmeans_clustering; pass False for "
                             "the literal raw-Y reading")

    # Frozen configuration (--stage fixed)
    parser.add_argument("--ka", type=int, default=None, help="K_a for --stage fixed")
    parser.add_argument("--m", type=int, default=None, help="m for --stage fixed")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)

    # Bookkeeping
    parser.add_argument("--out_dir", type=str, default=os.path.join("results", "grid_search"),
                        help="Where the checkpoint / result JSON is written")
    parser.add_argument("--no-resume", action="store_true", default=False,
                        help="Ignore an existing checkpoint and start over")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Only print the candidate ranges")
    parser.add_argument("--n-clusters", type=int, default=None, dest="n_clusters",
                        help="Cluster count K, used by --dry-run to expand the grid")
    parser.add_argument("--ka-ratios", type=str, default=None, dest="ka_ratios",
                        help="Comma-separated K_a/K ratios overriding the K-dependent "
                             "range, e.g. --ka-ratios 1 to lock K_a = K")
    parser.add_argument("--m-values", type=str, default=None, dest="m_values",
                        help="Comma-separated m values overriding the K-dependent "
                             "range, e.g. --m-values 2,3,4")
    parser.add_argument("--eval-final", action="store_true", default=False,
                        help="After selection, run the frozen configuration WITH labels")
    parser.add_argument("--eval-seeds", type=str, default="1,2,3,4,5,6,7,8,9,42",
                        help="Outer seeds for --eval-final")
    parser.add_argument("--times_clustering", type=int, default=10,
                        help="K-means repetitions used by the labelled evaluation")
    return parser.parse_args()


def dry_run(args):
    """Print the candidate ranges without loading the dataset."""
    print("=" * 78)
    print("BGLR grid-search candidate ranges (dry run)")
    print("=" * 78)
    print(f"  Stage 1, small K (K <= {SMALL_K_THRESHOLD}) : "
          f"K_a/K in {KA_RATIOS_SMALL_K}, m in {M_VALUES_SMALL_K}")
    print(f"  Stage 1, large K (K >  {SMALL_K_THRESHOLD}) : "
          f"K_a/K in {KA_RATIOS_LARGE_K}, m in {M_VALUES_LARGE_K}")
    if args.ka_ratios:
        print(f"  --ka-ratios override         : K_a/K in {parse_ka_ratios(args.ka_ratios)}")
    if args.m_values:
        print(f"  --m-values override          : m in {parse_int_list(args.m_values)}")
    print(f"  Stage 2, coefficients        : alpha,gamma in {ALPHA_GAMMA_GRID}, "
          f"beta in {BETA_GRID}  ({len(stage2_candidates())} candidates)")

    if args.n_clusters is None:
        print("\n  Pass --n-clusters K to expand the concrete stage-1 grid.")
        return
    K = args.n_clusters
    ratios = parse_ka_ratios(args.ka_ratios) if args.ka_ratios else ka_ratios_for(K)
    m_values = parse_int_list(args.m_values) if args.m_values else m_values_for(K)
    cands = stage1_candidates(K, ratios, m_values)
    print(f"\n  Stage 1 grid for K={K} (ratio range {ratios}, "
          f"{len(cands)} candidates):")
    for ratio, m in cands:
        ka = int(round(ratio * K))
        print(f"    K_a/K={ratio:<3g} K_a={ka:<6} m={m:<3} M={ka * m}")


def main():
    args = parse_args()

    if args.dry_run:
        return dry_run(args)

    from run_clustering import load_mat_dataset
    X, true_labels, aug_fea = load_mat_dataset(args.dataset, args.data_dir)
    n = X[0].shape[1]
    n_clusters = len(np.unique(true_labels))
    ratios = (parse_ka_ratios(args.ka_ratios) if args.ka_ratios
              else ka_ratios_for(n_clusters))
    m_values = (parse_int_list(args.m_values) if args.m_values
                else m_values_for(n_clusters))

    print("=" * 78)
    print(f"BGLR two-stage label-free grid search — {args.dataset}")
    print("=" * 78)
    print(f"  samples n        : {n}")
    print(f"  clusters K       : {n_clusters}")
    print(f"  views            : {len(X)} (+{len(aug_fea)} augmented)")
    print(f"  device           : {args.device}")
    print(f"  selection seed   : {args.select_seed}")
    print(f"  K-means reps     : {args.reps}")
    print(f"  Y convention     : "
          f"{'L2-normalized rows' if args.normalize_y else 'raw'}")
    k_note = "" if args.ka_ratios else (
        f"  (K {'<=' if n_clusters <= SMALL_K_THRESHOLD else '>'} {SMALL_K_THRESHOLD})")
    print(f"  ratio range      : K_a/K in {ratios}{k_note}")

    print(f"  m range          : {m_values}")
    if args.stage in ("1", "all", "2"):
        cand1 = stage1_candidates(n_clusters, ratios, m_values)
        print(f"  stage 1 grid     : {len(cand1)} anchor-structure candidates")
    if args.stage in ("2", "all"):
        print(f"  stage 2 grid     : {len(stage2_candidates())} coefficient candidates")

    checkpoint_path = os.path.join(args.out_dir, f"{args.dataset}_grid_search.json")
    if args.no_resume:
        records = []
    else:
        records = load_checkpoint(checkpoint_path)
        if records:
            print(f"\n  resuming from {checkpoint_path} ({len(records)} records)")
    done = {r["key"] for r in records if r.get("ok")}

    # ---------------- Stage 1 ----------------
    best1 = best_from_records(records, 1)
    if args.stage in ("1", "all"):
        print(f"\nStage 1 — anchor structure (alpha=beta=gamma="
              f"{STAGE1_ALPHA:g})")
        cands = []
        for ratio, m in stage1_candidates(n_clusters, ratios, m_values):
            # Keep the same visit order on resume by rebuilding the full grid
            cands.append({"ka": int(round(ratio * n_clusters)), "m": m,
                          "alpha": STAGE1_ALPHA, "beta": STAGE1_BETA,
                          "gamma": STAGE1_GAMMA})
        run_candidates(1, cands, X, aug_fea, n_clusters, args,
                       records, done, checkpoint_path)
        best1 = print_stage_summary(records, 1, "Stage 1 results",
                                    keys={rec_key(1, c) for c in cands})

    # ---------------- Stage 2 ----------------
    best2 = None
    if args.stage in ("2", "all"):
        if args.ka is not None and args.m is not None:
            ka, m = args.ka, args.m
            print(f"\nStage 2 — trade-off coefficients "
                  f"(K_a={ka}, m={m} given on the command line)")
        else:
            if best1 is None:
                best1 = best_from_records(records, 1)
            if best1 is None:
                raise SystemExit(
                    "Stage 2 needs a stage-1 result: run --stage 1 first, "
                    "or pass --ka/--m explicitly."
                )
            ka, m = best1["params"]["ka"], best1["params"]["m"]
            print(f"\nStage 2 — trade-off coefficients "
                  f"(frozen K_a={ka}, m={m}, M={ka * m})")
        cands = [{"ka": ka, "m": m, "alpha": a, "beta": b, "gamma": g}
                 for a, b, g in stage2_candidates()]
        run_candidates(2, cands, X, aug_fea, n_clusters, args,
                       records, done, checkpoint_path)
        best2 = print_stage_summary(records, 2, "Stage 2 results",
                                    keys={rec_key(2, c) for c in cands})

    # ---------------- Frozen / single configuration ----------------
    if args.stage == "fixed":
        missing = [k for k in ("ka", "m", "alpha", "beta", "gamma")
                   if getattr(args, k) is None]
        if missing:
            raise SystemExit(f"--stage fixed requires: {', '.join('--' + k for k in missing)}")
        cfg = {"ka": args.ka, "m": args.m, "alpha": args.alpha,
               "beta": args.beta, "gamma": args.gamma}
        print(f"\nFrozen configuration: {cfg_key(cfg)}")
        rec = evaluate_candidate(X, aug_fea, n_clusters, cfg, args)
        rec["stage"] = "fixed"
        rec["key"] = rec_key("fixed", cfg)
        records.append(rec)
        save_checkpoint(checkpoint_path, records)
        print(f"  SC={rec['score']:.4f}  iters={rec['iterations']}"
              f"  w=[{'/'.join(f'{w:.3f}' for w in rec['view_weights'])}]")
        best2 = rec

    # ---------------- Report ----------------
    print_final(best1, best2, n_clusters)
    print(f"\nCheckpoint: {checkpoint_path}")

    if args.eval_final:
        chosen = best2 if best2 is not None else best1
        if chosen is None:
            raise SystemExit("Nothing selected; cannot evaluate.")
        evaluate_frozen(X, true_labels, aug_fea, n_clusters,
                        chosen["params"], args)


if __name__ == "__main__":
    main()
