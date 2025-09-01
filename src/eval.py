# src/eval.py
import argparse, os, re, math, random
import numpy as np
import pandas as pd
from typing import List, Tuple, Set
from sklearn.metrics import silhouette_score
from scipy import sparse

ART = "artifacts"

# --- utils ---
def tokenize_skills(s: str) -> Set[str]:
    if not isinstance(s, str):
        return set()
    parts = re.split(r"[;,/|]", s.lower())
    return {p.strip() for p in parts if p.strip()}

def dcg(rels: List[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

def ndcg_at_k(rels: List[int], k: int) -> float:
    rels_k = rels[:k]
    idcg = dcg([1]*min(k, sum(rels_k)))
    return (dcg(rels_k)/idcg) if idcg > 0 else 0.0

# --- load artifacts consistent with pipeline ---
def load_env(cfg_path="config/config.yaml"):
    from src.utils.config import load_config
    cfg = load_config(cfg_path)
    courses = pd.read_parquet(os.path.join(ART, "courses.parquet"))
    if cfg.use_sbert:
        X = np.load(os.path.join(ART, "X_sbert.npy"))
    else:
        from src.utils.io import load_sparse_matrix
        X = load_sparse_matrix(os.path.join(ART, "X_tfidf.npz"))
    return cfg, courses, X

def row_sims(i: int, X):
    if sparse.issparse(X):
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(X[i], X).ravel()
    else:
        sims = X[i] @ X.T
    sims[i] = -1.0
    return sims

# --- EVAL: category / skills ---
def eval_by_category(courses: pd.DataFrame, X, topk=10, sample=300, seed=42):
    rng = random.Random(seed)
    idx_pool = [i for i in range(len(courses))
                if isinstance(courses.loc[i, "category"], str) and courses.loc[i, "category"].strip()]
    if not idx_pool:
        print("No 'category' values to evaluate.")
        return 0,0,0
    picks = rng.sample(idx_pool, min(sample, len(idx_pool)))
    precs, recs, ndcgs = [], [], []
    for i in picks:
        cat = str(courses.loc[i, "category"]).strip().lower()
        rel_set = {j for j in idx_pool if j != i and str(courses.loc[j, "category"]).strip().lower() == cat}
        if not rel_set:
            continue
        sims = row_sims(i, X)
        top_idx = np.argpartition(-sims, range(min(topk, len(sims))))[:topk]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        rels = [1 if int(j) in rel_set else 0 for j in top_idx]
        hits = sum(rels)
        precs.append(hits / topk)
        recs.append(hits / len(rel_set))
        ndcgs.append(ndcg_at_k(rels, topk))
    return float(np.mean(precs or [0])), float(np.mean(recs or [0])), float(np.mean(ndcgs or [0]))

def eval_by_skills(courses: pd.DataFrame, X, topk=10, sample=300, seed=42):
    rng = random.Random(seed)
    skill_lists = [tokenize_skills(s) for s in courses["skills"].astype(str).tolist()]
    idx_pool = [i for i, S in enumerate(skill_lists) if len(S) > 0]
    if not idx_pool:
        print("No 'skills' values to evaluate.")
        return 0,0,0
    picks = rng.sample(idx_pool, min(sample, len(idx_pool)))
    precs, recs, ndcgs = [], [], []
    for i in picks:
        Si = skill_lists[i]
        rel_set = {j for j, Sj in enumerate(skill_lists) if j != i and len(Si & Sj) > 0}
        if not rel_set:
            continue
        sims = row_sims(i, X)
        top_idx = np.argpartition(-sims, range(min(topk, len(sims))))[:topk]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        rels = [1 if int(j) in rel_set else 0 for j in top_idx]
        hits = sum(rels)
        precs.append(hits / topk)
        recs.append(hits / len(rel_set))
        ndcgs.append(ndcg_at_k(rels, topk))
    return float(np.mean(precs or [0])), float(np.mean(recs or [0])), float(np.mean(ndcgs or [0]))

# --- INTRINSIC ---
def eval_silhouette(courses: pd.DataFrame, X, sample=4000, metric="cosine", seed=42):
    if "cluster" not in courses.columns:
        print("No 'cluster' column. Run build first.")
        return 0.0
    rng = np.random.default_rng(seed)
    n = len(courses)
    take = min(sample, n)
    idx = rng.choice(n, size=take, replace=False)
    labels = courses["cluster"].to_numpy()[idx]
    from scipy import sparse as sp
    Xs = X[idx].toarray() if sp.issparse(X) else X[idx]
    try:
        score = silhouette_score(Xs, labels, metric=metric)
    except Exception:
        score = silhouette_score(Xs, labels, metric="euclidean")
    return float(score)

def eval_intra_inter(courses: pd.DataFrame, X, sample_per_cluster=50, seed=42):
    if "cluster" not in courses.columns:
        print("No 'cluster' column. Run build first.")
        return 0.0, 0.0
    rng = random.Random(seed)
    clusters = courses["cluster"].astype(int).tolist()
    by_c = {}
    for i, c in enumerate(clusters):
        by_c.setdefault(c, []).append(i)
    intra, inter = [], []
    for c, members in by_c.items():
        if len(members) < 2: continue
        picks = rng.sample(members, min(sample_per_cluster, len(members)))
        for i in picks:
            sims = row_sims(i, X)
            intra.extend([sims[j] for j in members if j != i])
            others = [j for j in range(len(courses)) if j not in members]
            other = rng.sample(others, min(50, len(others)))
            inter.extend([sims[j] for j in other])
    return float(np.mean(intra or [0])), float(np.mean(inter or [0]))

# --- CLI ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["category","skills","silhouette","intra_inter"], required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg, courses, X = load_env(args.config)

    if args.metric == "category":
        p, r, n = eval_by_category(courses, X, topk=args.topk, sample=args.sample)
        print(f"[Category] Precision@{args.topk}: {p:.3f}  Recall@{args.topk}: {r:.3f}  NDCG@{args.topk}: {n:.3f}")
    elif args.metric == "skills":
        p, r, n = eval_by_skills(courses, X, topk=args.topk, sample=args.sample)
        print(f"[Skills] Precision@{args.topk}: {p:.3f}  Recall@{args.topk}: {r:.3f}  NDCG@{args.topk}: {n:.3f}")
    elif args.metric == "silhouette":
        s = eval_silhouette(courses, X, sample=args.sample)
        print(f"[Clustering] Silhouette score (cosine): {s:.3f} (↑ tốt)")
    elif args.metric == "intra_inter":
        intra, inter = eval_intra_inter(courses, X)
        print(f"[Clusters] mean intra-sim: {intra:.3f}  |  mean inter-sim: {inter:.3f}  (intra >> inter là tốt)")

if __name__ == "__main__":
    main()
