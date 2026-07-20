"""Optional cross-encoder reranking of first-stage retrieval results.

A two-stage retriever: a fast ANN/cosine stage produces candidates, then a
cross-encoder (query, document) model rescores the shortlist for precision.
The model is loaded lazily and cached; if ``sentence-transformers`` is not
available the reranker degrades to a no-op that preserves the input order.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder  # lazy, heavy

    return CrossEncoder(model_name)


def rerank(
    query: str,
    documents: Sequence[str],
    indices: Sequence[int],
    base_scores: Sequence[float],
    top_k: int,
    model_name: str = DEFAULT_CROSS_ENCODER,
    alpha: float = 0.5,
) -> list[tuple[int, float]]:
    """Rerank ``documents`` for ``query`` and return ``(index, score)`` pairs.

    The final score blends the cross-encoder score with the first-stage score:
    ``alpha * ce + (1 - alpha) * base``. Falls back to the base ordering if the
    cross-encoder cannot be loaded.
    """
    if not documents:
        return []
    try:
        model = _load_cross_encoder(model_name)
    except Exception:
        order = np.argsort(-np.asarray(base_scores))[:top_k]
        return [(int(indices[i]), float(base_scores[i])) for i in order]

    pairs = [(query, doc) for doc in documents]
    ce_scores = np.asarray(model.predict(pairs), dtype=np.float32)
    # min-max normalize both signals so the blend is meaningful
    ce_norm = _minmax(ce_scores)
    base_norm = _minmax(np.asarray(base_scores, dtype=np.float32))
    blended = alpha * ce_norm + (1.0 - alpha) * base_norm
    order = np.argsort(-blended)[:top_k]
    return [(int(indices[i]), float(blended[i])) for i in order]


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
