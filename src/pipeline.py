import argparse
import os
import numpy as np
import pandas as pd
from typing import List
from .utils.config import load_config
from .utils.io import save_pickle, save_sparse_matrix, load_sparse_matrix, save_parquet
from .models.similarity import build_corpus, build_similarity_artifacts, topk_similar_by_index
from .models.clustering import fit_kmeans, compute_umap_2d
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ART = "artifacts"

def load_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError("Unsupported dataset format. Use CSV or Parquet.")

def get_text_matrix(cfg, corpus: List[str]):
    if cfg.use_sbert:
        X = np.load(os.path.join(ART, "X_sbert.npy"))
    else:
        from .utils.io import load_pickle
        vec = load_pickle(os.path.join(ART, "tfidf_vectorizer.pkl"))
        X = vec.transform(corpus)
    return X

def build(cfg_path: str, data_path: str):
    cfg = load_config(cfg_path)
    df = load_dataset(data_path)

    # Normalize columns: create missing columns if not exists
    for col in [cfg.columns.title, cfg.columns.description]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset. Edit config/config.yaml to match your CSV.")
    # Optional columns: if missing, create empty
    optional = ["skills","category","level","rating","url"]
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
    # Build a normalized dataframe with canonical columns
    norm = pd.DataFrame()
    for k, v in map_cols.items():
        if v and v in df.columns:
            norm[k] = df[v]
        else:
            norm[k] = None

    # If no id column, create one
    if norm["id"].isna().all():
        norm["id"] = range(len(df))

    # Build text corpus
    # translate cfg.text_fields (canonical keys) back to source columns
    field_map = { "title": "title", "description": "description", "skills": "skills",
                  "category": "category", "level": "level" }
    text_fields = [field_map.get(f, f) for f in cfg.text_fields]
    clean_df, corpus = build_corpus(norm, text_fields, cfg.min_characters)

    # Persist similarity artifacts
    build_similarity_artifacts(clean_df, corpus, cfg.use_sbert, cfg.sbert_model)

    # Create vector matrix for clustering & 2D emb
    if cfg.use_sbert:
        X = np.load(os.path.join(ART, "X_sbert.npy"))
    else:
        from .utils.io import load_pickle
        vec = load_pickle(os.path.join(ART, "tfidf_vectorizer.pkl"))
        X = vec.transform(corpus)

    # Fit KMeans
    km = fit_kmeans(X, cfg.kmeans_k, cfg.random_state)
    save_pickle(km, os.path.join(ART, "kmeans.pkl"))

    # UMAP 2D embedding for visualization
    emb = compute_umap_2d(X, cfg.random_state)
    np.save(os.path.join(ART, "umap_embedding.npy"), emb)

    # Save a "courses" parquet for the app
    clean_df = clean_df.reset_index(drop=True)
    clean_df["cluster"] = km.labels_
    save_parquet(clean_df, os.path.join(ART, "courses.parquet"))

    print("✅ Build complete. Artifacts saved to ./artifacts")

def query_similar(cfg_path: str, query_text: str, top_k: int = None):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from .utils.io import load_pickle, load_sparse_matrix

    cfg = load_config(cfg_path)
    courses = pd.read_parquet(os.path.join(ART, "courses.parquet"))
    texts = (
        courses["title"].fillna("") + " | " +
        courses["description"].fillna("") + " | " +
        courses["skills"].fillna("")
    ).str.lower().tolist()

    if cfg.use_sbert:
        from sentence_transformers import SentenceTransformer
        model_name = load_pickle(os.path.join(ART, "sbert_model_name.pkl"))
        model = SentenceTransformer(model_name)
        X = np.load(os.path.join(ART, "X_sbert.npy"))
        q = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        sims = (q @ X.T).ravel()
    else:
        vec = load_pickle(os.path.join(ART, "tfidf_vectorizer.pkl"))
        X = load_sparse_matrix(os.path.join(ART, "X_tfidf.npz"))
        q = vec.transform([query_text])
        sims = cosine_similarity(q, X).ravel()

    # --- Hard filter theo từ khóa cốt lõi (giảm nhiễu) ---
    import re
    tokens = set(re.findall(r"[a-zA-Z]+", query_text.lower()))
    must_vocab = {"python","pytorch","tensorflow","machine","learning","deep","dl","ai","data"}
    must = tokens & must_vocab
    if must:
        mask = pd.Series(texts).str.contains("|".join(sorted(must)), regex=True)
        if mask.sum() >= 5:
            sims[~mask.values] = -1.0

    k = top_k or cfg.top_k

    # --- (Optional) Rerank bằng CrossEncoder cho top-50 ---
    k0 = max(k, 50)
    idx0 = np.argpartition(-sims, range(min(k0, len(sims))))[:k0]
    idx0 = idx0[np.argsort(-sims[idx0])]

    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query_text, texts[i]) for i in idx0]
        rerank = ce.predict(pairs)
        order = np.argsort(-rerank)[:k]
        final_idx = idx0[order]
        final_score = 0.5 * rerank[order] + 0.5 * sims[final_idx]
        return courses.iloc[final_idx].assign(score=final_score)
    except Exception:
        # Fallback: không rerank
        idx = idx0[:k]
        return courses.iloc[idx].assign(score=sims[idx])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build","query"], default="build")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default="data/Coursera.csv")
    parser.add_argument("--text", default=None, help="free-text query when mode=query")
    parser.add_argument("--topk", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "build":
        build(args.config, args.data)
    elif args.mode == "query":
        assert args.text, "--text is required for query mode"
        res = query_similar(args.config, args.text, args.topk)
        print(res[["id","title","category","level","rating","score"]].head(20).to_string(index=False))
