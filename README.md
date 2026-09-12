# BGLR: Learning Manifold-Consistent Anchor Bases for Multi-View Short Text Clustering

Reference implementation of BGLR, a multi-view short text clustering framework.
It takes three fixed text-derived views (TF-IDF + SVD, MPNet on the original
text, MPNet on a semantic rewrite), learns view-specific anchor bases together
with simplex-constrained sample-to-anchor coordinates under a shared orthogonal
clustering embedding, and returns the final partition by K-means on the fused
representation. No text encoder is trained or fine-tuned.

## Requirements

Python >= 3.8:

```bash
pip install -r requirements.txt
```

GPU acceleration is optional; pass `--device cuda` to use it.

## Code layout

| Path | Role |
| --- | --- |
| `run_clustering.py` | Main entry: build features (`--mode preprocess`) and run clustering (`--mode cluster`) |
| `run_grid_search.py` | Two-stage label-free parameter selection (silhouette criterion, no labels used) |
| `Generate_Synonym.py` | Builds the semantic-rewrite view with a local LLM served through an OpenAI-compatible API |
| `bglr/core.py` | `BGLR` model: fit and evaluation |
| `bglr/algo_qp.py` | Stabilized ADMM solver |
| `bglr/fast_multi_clr.py` | Anchor selection and bipartite graph construction |
| `bglr/manifold.py` | Sparse TF-IDF graph Laplacian |
| `bglr/preprocess.py` | CSV to `.mat` view builder |
| `bglr/metrics.py` | ACC / NMI / purity / F-score / silhouette |
| `bglr/utils.py` | Shared math helpers |

## Usage

0. (optional) Build the rewrite column for another corpus. The raw corpora
   under `data/` hold one short text per line; this step needs a local
   OpenAI-compatible server (e.g. vLLM serving `Qwen/Qwen2.5-0.5B-Instruct`):

   ```bash
   python Generate_Synonym.py --input data/agnews --output data/Augment/agnews.csv
   ```

1. Build the views from a CSV with columns `label, text, text1`, where `text1`
   is the rewritten text (additional columns are ignored). `tweet` is shipped
   as a worked example:

   ```bash
   python run_clustering.py --mode preprocess --csv data/Augment/tweet.csv --dataset tweet
   ```

2. Run clustering with an explicit configuration:

   ```bash
   python run_clustering.py --mode cluster --dataset tweet \
       --k2 89 --m 2 --alpha 0.01 --beta 100 --gamma 10 --device cuda
   ```

3. Or select the configuration without touching any labels:

   ```bash
   python run_grid_search.py --dataset tweet --device cuda
   ```

Each script prints its full option list with `--help`.

## Data format

`datasets/<name>.mat` packs the three views:

| Key | Content |
| --- | --- |
| `fea` | 1x3 cell: view 1 (TF-IDF + SVD), view 2 (MPNet, original text); slot 3 holds a zero placeholder that `aug_fea` replaces at run time |
| `aug_fea` | 1x1 cell: view 3 (MPNet, rewritten text) |
| `Y`, `gnd` | `(n, 1)` ground-truth labels |

Feature matrices are stored as `d x n` (one column per short text). The raw
corpora and the `tweet` example live in `data/`; the `.mat` views under
`datasets/` are generated and not tracked, so run `--mode preprocess` first.

## License

Released under the MIT License; see `LICENSE`.
