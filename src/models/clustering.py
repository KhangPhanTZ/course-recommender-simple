import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.cluster import KMeans
import umap
from ..utils.io import save_pickle
from scipy import sparse

def fit_kmeans(X, k: int, random_state: int = 42) -> KMeans:
    if sparse.issparse(X):
        X_dense = X  # KMeans in sklearn now supports CSR for some versions; else convert
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            km.fit(X_dense)
            return km
        except Exception:
            X_dense = X.toarray()
    else:
        X_dense = X
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    km.fit(X_dense)
    return km

def compute_umap_2d(X, random_state: int = 42, n_neighbors: int = 15, min_dist: float = 0.1):
    if sparse.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = X
    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist)
    emb = reducer.fit_transform(X_dense)
    return emb
