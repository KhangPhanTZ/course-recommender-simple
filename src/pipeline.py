"""End-to-end build pipeline for the course recommender.

Produces a coherent, backend-agnostic artifact set that the serving layer
(:mod:`src.recsys.recommender`) and the API load through the storage
abstraction (local filesystem or S3):

    meta.json            backend, embedding dim, model name, row count
    courses.parquet      cleaned dataset + cluster labels
    embeddings.npy       dense SBERT vectors            (sbert backend)
    vector.index         serialized FAISS/numpy index   (sbert backend)
    tfidf_vectorizer.pkl fitted TF-IDF vectorizer        (tfidf backend)
    X_tfidf.npz          sparse TF-IDF matrix            (tfidf backend)
    kmeans.pkl           KMeans model                    (clusters)

Run::

    python -m src.pipeline --mode build --data data/Coursera.csv
    python -m src.pipeline --mode query --text "deep learning with pytorch"
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle

import numpy as np
import pandas as pd

from .data import discover, load_catalog, source_counts
from .models.clustering import fit_kmeans
from .models.similarity import build_corpus, embed_sbert, fit_tfidf
from .storage import ArtifactStore, get_artifact_store
from .utils.config import Config, load_config


# ------------------------------------------------------------------ helpers
def _np_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _parquet_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _sparse_to_bytes(mat) -> bytes:
    from scipy import sparse

    buf = io.BytesIO()
    sparse.save_npz(buf, mat)
    return buf.getvalue()


def load_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError("Unsupported dataset format. Use CSV or Parquet.")


def _normalize_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Map source columns onto canonical names; create missing optional ones."""
    for col in [cfg.columns.title, cfg.columns.description]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in dataset. Edit config/config.yaml."
            )
    map_cols = {
        "id": cfg.columns.id,
        "title": cfg.columns.title,
        "description": cfg.columns.description,
        "skills": cfg.columns.skills,
        "category": cfg.columns.category,
        "level": cfg.columns.level,
        "rating": cfg.columns.rating,
        "url": cfg.columns.url,
    }
    norm = pd.DataFrame()
    for canonical, source in map_cols.items():
        norm[canonical] = df[source] if source and source in df.columns else None
    if norm["id"].isna().all():
        norm["id"] = range(len(df))
    return norm


# -------------------------------------------------------------------- build
def build(
    cfg_path: str = "config/config.yaml",
    data_path: str = "data/Coursera.csv",
    store: ArtifactStore | None = None,
) -> None:
    cfg = load_config(cfg_path)
    store = store or get_artifact_store()

    # A directory triggers multi-platform ingestion: every recognized Coursera /
    # Udemy / edX file inside is normalized onto one schema and merged. A single
    # file keeps the legacy config-driven column mapping (back-compat).
    sources: dict[str, int] = {}
    if os.path.isdir(data_path):
        norm = load_catalog(discover(data_path))
        sources = source_counts(norm)
        print(f"Merged {len(norm)} courses from {len(sources)} source(s): {sources}")
    else:
        norm = _normalize_columns(load_dataset(data_path), cfg)

    clean_df, corpus = build_corpus(norm, cfg.text_fields, cfg.min_characters)
    clean_df = clean_df.reset_index(drop=True)

    meta = {
        "backend": cfg.backend,
        "n_rows": int(len(clean_df)),
        "text_fields": cfg.text_fields,
    }
    if sources:
        meta["sources"] = sources

    # --- embeddings + retrieval index ---
    if cfg.use_sbert:
        model, X = embed_sbert(corpus, cfg.sbert_model)
        X = np.asarray(X, dtype=np.float32)
        store.write_bytes("embeddings.npy", _np_to_bytes(X))

        from .recsys.index import VectorIndex

        index = VectorIndex(dim=X.shape[1]).add(X)
        store.write_bytes("vector.index", index.to_bytes())
        meta["dim"] = int(X.shape[1])
        meta["sbert_model"] = cfg.sbert_model
        cluster_input = X
    else:
        vec, X = fit_tfidf(corpus, max_features=cfg.tfidf_max_features)
        store.write_bytes("tfidf_vectorizer.pkl", pickle.dumps(vec))
        store.write_bytes("X_tfidf.npz", _sparse_to_bytes(X))
        meta["dim"] = int(X.shape[1])
        cluster_input = X

    # --- clustering ---
    km = fit_kmeans(cluster_input, cfg.kmeans_k, cfg.random_state)
    store.write_bytes("kmeans.pkl", pickle.dumps(km))
    clean_df["cluster"] = km.labels_
    meta["kmeans_k"] = int(cfg.kmeans_k)

    # --- persist dataset + metadata ---
    store.write_bytes("courses.parquet", _parquet_to_bytes(clean_df))
    store.write_bytes("meta.json", json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))

    print(f"✅ Build complete ({meta['backend']} backend, {meta['n_rows']} courses). "
          f"Artifacts written via {store.__class__.__name__}.")


# -------------------------------------------------------------------- query
def query_similar(cfg_path: str, query_text: str, top_k: int | None = None):
    """CLI/back-compat helper returning a DataFrame of recommendations."""
    from .recsys.recommender import Recommender

    cfg = load_config(cfg_path)
    rec = Recommender.load(get_artifact_store())
    results = rec.recommend(
        query_text,
        top_k=top_k or cfg.top_k,
        use_rerank=cfg.use_rerank,
        overfetch=cfg.overfetch,
    )
    return pd.DataFrame([r.to_dict() for r in results])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "query"], default="build")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default="data/Coursera.csv")
    parser.add_argument("--text", default=None, help="free-text query when mode=query")
    parser.add_argument("--topk", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "build":
        build(args.config, args.data)
    else:
        assert args.text, "--text is required for query mode"
        res = query_similar(args.config, args.text, args.topk)
        cols = [c for c in ["id", "title", "category", "level", "rating", "score"] if c in res.columns]
        print(res[cols].head(20).to_string(index=False))
