"""Offline evaluation of the recommender — retrieval quality, latency, clusters.

Content-based systems have no click logs, so relevance is proxied by the
catalog's own **category** labels via leave-one-out: for each sampled course we
retrieve its nearest neighbours and treat same-category neighbours as relevant.
This yields honest, reproducible numbers (Precision@k, Recall@k, MRR, nDCG@k,
hit-rate) that move with retrieval quality.

``evaluate`` also benchmarks query latency (p50/p95) through the real
``recommend`` path and reports the clustering silhouette. Everything degrades
gracefully when a signal is missing (no category column, single cluster, …).
"""
from __future__ import annotations

import math
import random
import time
from datetime import UTC, datetime
from statistics import mean
from typing import Any

import numpy as np

# A fixed probe set so latency numbers are comparable across runs.
DEFAULT_QUERIES = [
    "deep learning with pytorch",
    "sql for data analysis",
    "become an ml engineer",
    "cloud computing on aws",
    "product management and agile",
    "test automation with selenium",
    "data engineering with spark",
    "nlp and transformers",
]


def _ndcg(rels: list[int], k: int) -> float:
    rels_k = rels[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels_k))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, sum(rels_k))))
    return dcg / ideal if ideal > 0 else 0.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return round(s[idx], 2)


def _retrieval_metrics(rec, k: int, sample: int, seed: int) -> dict[str, Any]:
    df = rec.courses
    if "category" not in df.columns:
        return {}
    cats = df["category"].astype("string").fillna("").str.strip().str.lower()
    counts = cats.value_counts().to_dict()
    pool = [i for i in range(rec.size) if cats.iloc[i] and counts.get(cats.iloc[i], 0) > 1]
    if not pool:
        return {}

    rng = random.Random(seed)
    picks = rng.sample(pool, min(sample, len(pool)))
    precision, recall, mrr, ndcg, hit = [], [], [], [], []
    for i in picks:
        cat = cats.iloc[i]
        neighbours = rec.similar_to(df.iloc[i]["id"], top_k=k)
        rels = [1 if str(nb.category or "").strip().lower() == cat else 0 for nb in neighbours]
        found = sum(rels)
        total_rel = counts[cat] - 1  # exclude self
        precision.append(found / len(rels) if rels else 0.0)
        recall.append(found / min(total_rel, k) if total_rel > 0 else 0.0)
        hit.append(1.0 if found else 0.0)
        rank = next((j + 1 for j, r in enumerate(rels) if r), 0)
        mrr.append(1.0 / rank if rank else 0.0)
        ndcg.append(_ndcg(rels, k))

    return {
        "n_eval": len(picks),
        "precision_at_k": round(mean(precision), 4),
        "recall_at_k": round(mean(recall), 4),
        "hit_rate": round(mean(hit), 4),
        "mrr": round(mean(mrr), 4),
        "ndcg_at_k": round(mean(ndcg), 4),
    }


def _latency_ms(rec, k: int, queries: list[str]) -> dict[str, float]:
    times: list[float] = []
    for q in queries:
        start = time.perf_counter()
        rec.recommend(q, top_k=k)
        times.append((time.perf_counter() - start) * 1000)
    return {"p50": _percentile(times, 50), "p95": _percentile(times, 95), "mean": round(mean(times), 2)}


def _silhouette(rec, sample: int, seed: int) -> float | None:
    df = rec.courses
    if "cluster" not in df.columns:
        return None
    labels = df["cluster"].to_numpy()
    if len(set(labels.tolist())) < 2:
        return None
    X = rec._embeddings if rec.backend == "sbert" else rec._tfidf_mat
    if X is None:
        return None
    try:
        from sklearn.metrics import silhouette_score

        n = rec.size
        if n > sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=sample, replace=False)
            X, labels = X[idx], labels[idx]
        return round(float(silhouette_score(X, labels, metric="cosine")), 4)
    except Exception:
        return None


def evaluate(rec, k: int = 10, sample: int = 300, seed: int = 42,
             queries: list[str] | None = None) -> dict[str, Any]:
    """Run the full evaluation and return a JSON-serializable metrics dict."""
    metrics: dict[str, Any] = {
        "k": k,
        "backend": rec.backend,
        "n_courses": rec.size,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    metrics.update(_retrieval_metrics(rec, k, sample, seed))
    metrics["latency_ms"] = _latency_ms(rec, k, queries or DEFAULT_QUERIES)
    sil = _silhouette(rec, sample=min(1000, rec.size), seed=seed)
    if sil is not None:
        metrics["silhouette"] = sil
    if "sources" in rec.meta:
        metrics["sources"] = rec.meta["sources"]
    return metrics
