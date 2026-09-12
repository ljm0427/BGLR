"""
Text data preprocessor for multi-view text clustering.

Builds three views from a CSV file and serializes them into a MATLAB .mat file:
  View 1: original text               -> TF-IDF + SVD reduction
  View 2: original text               -> MPNet embedding (768-d)
  View 3: best augmented text (text1) -> MPNet embedding (768-d)
"""

import os
import numpy as np
import pandas as pd
from scipy.io import savemat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class TextDataPreprocessor:
    """
    Preprocess text CSV data for multi-view clustering.

    View 1: original text → TF-IDF + SVD reduction → L2 column normalization
    View 2: original text → MPNet-768 → L2 normalization
    View 3: best augmented text (text1) → MPNet-768 → L2 normalization (not learned)

    Parameters
    ----------
    tfidf_dim : int
        SVD reduced dimension for TF-IDF (default 256).
    tfidf_max_features : int
        Max TF-IDF features (default 5000).
    mpnet_model_name : str
        SentenceTransformer model name (default 'all-mpnet-base-v2').
    batch_size : int
        Batch size for MPNet encoding (default 32).
    """

    # Encodings tried in order when reading a CSV. Corpora that have passed
    # through external tooling sometimes carry a few bytes that are not valid
    # UTF-8 (e.g. an en dash truncated by a cp1252 round trip); a strict UTF-8
    # read would abort the whole preprocessing run, so we fall back in order and
    # warn, instead of failing.
    CSV_ENCODINGS = ('utf-8-sig', 'utf-8', 'latin-1')

    def __init__(
        self,
        tfidf_dim=256,
        tfidf_max_features=5000,
        mpnet_model_name='all-mpnet-base-v2',
        batch_size=32,
    ):
        self.tfidf_dim = tfidf_dim
        self.tfidf_max_features = tfidf_max_features
        self.mpnet_model_name = mpnet_model_name
        self.batch_size = batch_size

        self.vectorizer_ = None
        self.svd_ = None
        self.model_ = None

    @classmethod
    def _read_csv(cls, csv_path):
        """
        Read a CSV, trying :attr:`CSV_ENCODINGS` in order.

        Only decoding failures trigger the fallback; any other error (missing
        file, malformed CSV, ...) propagates unchanged. When a fallback happens
        the affected character count is reported, so that a genuinely corrupted
        corpus stays visible instead of being silently accepted.

        Returns
        -------
        df : pandas.DataFrame
        encoding : str, the encoding that worked
        """
        decode_err = None
        for encoding in cls.CSV_ENCODINGS:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
            except UnicodeError as err:
                decode_err = decode_err or err
                continue

            if encoding != cls.CSV_ENCODINGS[0]:
                with open(csv_path, 'rb') as fh:
                    n_bad = fh.read().decode('utf-8', errors='replace').count('\ufffd')
                print(f"[WARN] {csv_path} is not valid {cls.CSV_ENCODINGS[0]}: {decode_err}")
                print(f"[WARN] Falling back to encoding='{encoding}'; "
                      f"{n_bad} character(s) are affected and non-ASCII text "
                      f"may be mangled.")
            return df, encoding

        raise ValueError(
            f"Could not decode {csv_path} with any of {cls.CSV_ENCODINGS}: {decode_err}"
        )

    def load_csv(self, csv_path):
        """
        Load a CSV whose first three columns are label, text and text1, where
        text1 is the retained semantic rewrite. Any additional columns are
        ignored.

        Returns
        -------
        original_texts : list of str
        best_aug_texts : list of str (text1 only)
        labels : ndarray (n,) 0-based
        n_clusters : int
        """
        df, encoding = self._read_csv(csv_path)
        cols = df.columns.tolist()
        print(f"[INFO] CSV columns: {cols} (encoding: {encoding})")

        if len(cols) < 3:
            raise ValueError(
                f"CSV needs at least 3 columns (label, text, text1), got {len(cols)}: {cols}"
            )

        label_col = cols[0]
        text_col = cols[1]
        best_aug_col = cols[2]

        labels = df[label_col].values.astype(np.int64)
        if labels.min() >= 1:
            labels = labels - labels.min()

        original_texts = df[text_col].fillna("").astype(str).tolist()
        best_aug_texts = df[best_aug_col].fillna("").astype(str).tolist()

        n_clusters = len(np.unique(labels))
        print(f"[INFO] Samples: {len(labels)}, Clusters: {n_clusters}")
        print(f"[INFO] Original text avg length: {np.mean([len(t.split()) for t in original_texts]):.1f} words")
        avg_len = np.mean([len(t.split()) for t in best_aug_texts])
        print(f"[INFO] Best augmented '{best_aug_col}' avg length: {avg_len:.1f} words")

        return original_texts, best_aug_texts, labels, n_clusters

    def extract_tfidf_svd(self, texts):
        """TF-IDF → SVD → L2 column normalization → (d, n)."""
        print(f"[TF-IDF] Processing {len(texts)} texts...")

        self.vectorizer_ = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            stop_words='english',
            sublinear_tf=True,
            max_df=0.8,
            min_df=2,
        )
        tfidf = self.vectorizer_.fit_transform(texts)
        print(f"  TF-IDF shape: {tfidf.shape}")

        n_comp = min(self.tfidf_dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        n_comp = max(n_comp, 10)
        self.svd_ = TruncatedSVD(n_components=n_comp, random_state=42)
        reduced = self.svd_.fit_transform(tfidf)
        print(f"  SVD reduced: {reduced.shape}, explained variance: {self.svd_.explained_variance_ratio_.sum():.3f}")

        # L2 column normalization → (d, n)
        result = normalize(reduced, norm='l2', axis=1).T.astype(np.float64)
        print(f"  View1 (TF-IDF+SVD): {result.shape} (d×n)")
        return result

    def extract_mpnet(self, texts):
        """MPNet-768 embedding → L2 normalization → (768, n)."""
        if self.model_ is None:
            from sentence_transformers import SentenceTransformer
            print(f"[MPNet] Loading model: {self.mpnet_model_name} ...")
            self.model_ = SentenceTransformer(self.mpnet_model_name)

        embeddings = self.model_.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )  # (n, 768)

        result = embeddings.T.astype(np.float64)  # → (768, n)
        print(f"  MPNet shape: {result.shape} (d×n)")
        return result

    def process(self, csv_path, output_dir='datasets', dataset_name='MyTextData'):
        """
        Full pipeline: CSV → features → .mat file.

        Returns
        -------
        mat_path : str, path to saved .mat file.
        """
        # 1. Load data (only text1 as the best augmented view)
        original_texts, best_aug_texts, labels, n_clusters = self.load_csv(csv_path)

        # 2. View 1: TF-IDF + SVD
        print("\n========== View 1: original text → TF-IDF + SVD ==========")
        view1 = self.extract_tfidf_svd(original_texts)

        # 3. View 2: MPNet on original text
        print("\n========== View 2: original text → MPNet-768 ==========")
        view2 = self.extract_mpnet(original_texts)

        # 4. Augmented view: MPNet on text1 (not learned)
        print("\n========== View 3 augmented: text1 → MPNet-768 ==========")
        aug_feature = self.extract_mpnet(best_aug_texts)

        # 5. Assemble .mat data
        n = len(original_texts)

        # fea: 1×3 cell
        fea = np.empty((3,), dtype=object)
        fea[0] = view1
        fea[1] = view2
        fea[2] = np.zeros((1, n))  # placeholder, filled at runtime

        # aug_fea: 1×1 cell (only the best augmented view)
        aug_fea = np.empty((1,), dtype=object)
        aug_fea[0] = aug_feature

        fea_names = np.empty((3,), dtype=object)
        fea_names[0] = "View1_TFIDF_SVD"
        fea_names[1] = "View2_MPNet_Original"
        fea_names[2] = "View3_MPNet_Augmented(text1)"

        # 6. Save
        os.makedirs(output_dir, exist_ok=True)
        mat_path = os.path.join(output_dir, f"{dataset_name}.mat")

        savemat(mat_path, {
            'fea': fea,
            'aug_fea': aug_fea,
            'Y': labels.astype(np.float64).reshape(-1, 1),
            'gnd': labels.astype(np.float64).reshape(-1, 1),
            'fea_names': fea_names,
        })

        print(f"\n{'='*60}")
        print(f"[DONE] Saved: {mat_path}")
        print(f"{'='*60}")
        for i in range(2):
            print(f"  {fea_names[i]}: {fea[i].shape[0]}×{fea[i].shape[1]} (d×n)")
        print(f"  {fea_names[2]}: {aug_feature.shape[0]}×{aug_feature.shape[1]} (d×n)")
        print(f"  Y/gnd: {n}×1, n_clusters={n_clusters}")

        return mat_path
