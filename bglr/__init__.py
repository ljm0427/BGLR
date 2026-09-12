"""
BGLR: Learning Manifold-Consistent Anchor Bases for Multi-View Short Text Clustering
"""

from .core import BGLR
from .utils import normalize_fea, opt_s, l2_distance
from .metrics import clustering_measure, kmeans_clustering
from .preprocess import TextDataPreprocessor

__version__ = "1.0.0"
__all__ = [
    "BGLR",
    "normalize_fea",
    "opt_s",
    "l2_distance",
    "clustering_measure",
    "kmeans_clustering",
    "TextDataPreprocessor",
]
