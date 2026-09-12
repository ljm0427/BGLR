"""
run_clustering.py — Main entry point for BGLR multi-view text clustering.

Usage:
    # Preprocess CSV → .mat
    python run_clustering.py --mode preprocess --csv data.csv --dataset MyTextData

    # Run clustering on preprocessed data
    python run_clustering.py --mode cluster --dataset MyTextData --device cpu

    # Run with GPU acceleration
    python run_clustering.py --mode cluster --dataset MyTextData --device cuda

    # Run both
    python run_clustering.py --mode all --csv data.csv --dataset MyTextData --device cuda
"""

import argparse
import os
import sys
import time
import numpy as np
from scipy.io import loadmat

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bglr import BGLR, TextDataPreprocessor


def load_mat_dataset(dataset_name, data_dir='datasets'):
    """Load .mat dataset and return X, true_labels, aug_fea."""
    mat_path = os.path.join(data_dir, f'{dataset_name}.mat')
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Dataset not found: {mat_path}. Run preprocess first.")

    print(f"Loading dataset: {mat_path}")
    data = loadmat(mat_path)

    # fea: 1×3 cell (object array)
    fea_raw = data['fea'][0]  # (3,) object array
    X = [fea_raw[i] for i in range(3)]

    # aug_fea: 1×1 cell (only the best augmented view, default text1)
    aug_raw = data['aug_fea'][0]
    aug_fea = [aug_raw[0]]  # single augmented view


    # Labels
    Y = data['Y'].flatten().astype(np.int64)
    if Y.min() >= 1:
        Y = Y - Y.min()

    return X, Y, aug_fea


def run_preprocess(args):
    """Run data preprocessing."""
    preprocessor = TextDataPreprocessor(
        tfidf_dim=args.tfidf_dim,
        mpnet_model_name=args.mpnet_model,
        batch_size=args.batch_size,
    )
    mat_path = preprocessor.process(
        csv_path=args.csv,
        output_dir=args.data_dir,
        dataset_name=args.dataset,
    )
    print(f"\nPreprocessing complete: {mat_path}")


def run_clustering(args):
    """Run BGLR clustering."""
    # Load data
    X, true_labels, aug_fea = load_mat_dataset(args.dataset, args.data_dir)

    n = X[0].shape[1]
    n_clusters = len(np.unique(true_labels))

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"  Samples: {n}, Clusters: {n_clusters}")
    for iv in range(3):
        print(f"  View {iv+1}: {X[iv].shape[0]}×{X[iv].shape[1]}")

    print(f"  Augmented views: {len(aug_fea)}")
    print(f"{'='*60}")

    # Configure the model with explicit parameters
    k = n_clusters
    k2_val = args.k2 if args.k2 is not None else 3 * k
    m_val = args.m if args.m is not None else 1

    model = BGLR(
        n_clusters=n_clusters,
        k2=k2_val,
        m=m_val,
        alpha=args.alpha if args.alpha is not None else 1.0,
        beta=args.beta if args.beta is not None else 1.0,
        gamma=args.gamma,
        k_nn=10,
        lap_knn=args.lap_knn,
        max_iter=200,
        tol=1e-6,
        device=args.device,
        seed=args.seed,
        verbose=not args.quiet,
        freeze_anchors=args.freeze_anchors,
        save_snapshots=args.save_snapshots,
        aug_target=None if args.aug_target < 0 else args.aug_target,
        aug_index=args.aug_index,
        num_views=args.views,
        adaptive_weight=args.adaptive_weight,
    )

    # Fit
    t_start = time.time()
    model.fit(X, true_labels=true_labels, aug_fea=aug_fea, times_clustering=args.times_clustering)
    t_total = time.time() - t_start

    # Report
    print(f"\nClustering Complete — {t_total:.1f}s")
    if model.params_:
        print(f"  Params: {model.params_}")
        print(f"  ACC: {model.acc_:.4f}")
        if model.silhouette_ is not None:
            print(f"  Silhouette: {model.silhouette_:.4f}")

    # Save convergence data
    if args.save_snapshots and model.snapshots_ is not None:
        conv_dir = os.path.join('results', args.dataset)
        os.makedirs(conv_dir, exist_ok=True)
        conv_path = os.path.join(conv_dir, 'convergence.npz')
        save_dict = {'obj_vals': np.array(model.obj_vals_)}
        for key, val in model.snapshots_.items():
            save_dict[f'Z_bar_{key}'] = val
        np.savez_compressed(conv_path, **save_dict)
        print(f"  Convergence data saved to: {conv_path}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description='Multi-view Text Clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--mode', type=str, default='cluster',
                        choices=['preprocess', 'cluster', 'all'],
                        help='Operation mode')

    # Data options
    parser.add_argument('--csv', type=str, default=None,
                        help='Input CSV file path (for preprocess mode)')
    parser.add_argument('--dataset', type=str, default='MyTextData',
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='datasets',
                        help='Data directory')

    # Preprocessing options
    parser.add_argument('--tfidf_dim', type=int, default=256,
                        help='TF-IDF SVD dimension')
    parser.add_argument('--mpnet_model', type=str, default='all-mpnet-base-v2',
                        help='SentenceTransformer model name')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='MPNet batch size')

    # Clustering options
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Computation device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--times_clustering', type=int, default=10,
                        help='K-means repetitions for evaluation')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Suppress progress output')

    # Explicit hyper-parameters (for label-free selection see run_grid_search.py)
    parser.add_argument('--k2', type=int, default=None,
                        help='Number of anchor-clusters (default: 3 x K)')
    parser.add_argument('--m', type=int, default=None,
                        help='Anchors per anchor-cluster (default: 1)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Anchor structure regularization weight')
    parser.add_argument('--beta', type=float, default=None,
                        help='Cluster consistency weight')

    # View / augmentation control
    parser.add_argument('--views', type=int, default=None,
                        help='Number of views to use (2 for 2-view, default=all)')
    parser.add_argument('--aug-target', type=int, default=2,
                        help='Which view (0-based) to replace with augmented data '
                             '(default=2, set to -1 to disable augmentation)')
    parser.add_argument('--aug-index', type=int, default=0,
                        help='Which augmented view to use (0-based, default=0). '
                             'Only used when augmentation is enabled.')

    # Text-specific regularization
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='Graph Laplacian regularization weight (default=0=disabled)')
    parser.add_argument('--lap-knn', type=int, default=10,
                        help='kNN parameter for Laplacian graph (default=10)')
    parser.add_argument('--adaptive-weight', type=lambda x: x.lower() != 'false', default=True,
                        help='Use adaptive view weights (default=True). Set to False for equal weights.')
    parser.add_argument('--freeze-anchors', action='store_true', default=False,
                        help='Ablation: freeze anchors at K-means initialization (w/o Learnable Anchors)')
    parser.add_argument('--save-snapshots', action='store_true', default=False,
                        help='Save intermediate Z_bar at iter 0,25,50,100 and final for convergence visualization.')

    args = parser.parse_args()

    if args.mode == 'preprocess':
        if args.csv is None:
            parser.error("--csv is required for preprocess mode")
        run_preprocess(args)

    elif args.mode == 'cluster':
        run_clustering(args)

    elif args.mode == 'all':
        if args.csv is None:
            parser.error("--csv is required for all mode")
        run_preprocess(args)
        run_clustering(args)


if __name__ == '__main__':
    main()
