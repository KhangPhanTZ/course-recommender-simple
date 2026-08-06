"""High-level recommendation service used by the API and CLI.

Loads the trained artifacts through the :mod:`src.storage` abstraction so the
identical code path serves from a laptop (local files) or from AWS
(S3 + optional FAISS). Supports two embedding backends:

- ``sbert``: dense Sentence-BERT vectors + FAISS/numpy ANN index (default,
  best semantic quality).
- ``tfidf``: sparse TF-IDF vectors + exact cosine (lightweight fallback).

The public surface is intentionally small:

    rec = Recommender.load(store)
    rec.recommend("deep learning with pytorch", top_k=10, filters={"level": "beginner"})
    rec.similar_to(course_id=42, top_k=5)
"""
from __future__ import annotations

import io
import json
import pickle
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..storage import ArtifactStore, get_artifact_store
from .index import VectorIndex

META_KEY = "meta.json"
COURSES_KEY = "courses.parquet"
EMB_KEY = "embeddings.npy"
INDEX_KEY = "vector.index"
TFIDF_VEC_KEY = "tfidf_vectorizer.pkl"
TFIDF_MAT_KEY = "X_tfidf.npz"


@dataclass
class Recommendation:
    """A single scored recommendation returned to callers."""

    id: Any
    title: str
    score: float
    category: str | None = None
    level: str | None = None
    rating: Any | None = None
    url: str | None = None
    skills: str | None = None
    explanation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "score": round(float(self.score), 4),
            "category": self.category,
            "level": self.level,
            "rating": self.rating,
            "url": self.url,
            "skills": self.skills,
            "explanation": self.explanation,
        }


@dataclass
class Recommender:
    """In-memory recommender backed by artifacts from an :class:`ArtifactStore`."""

    courses: pd.DataFrame
    backend: str
    meta: dict[str, Any]
    store: ArtifactStore
    _index: VectorIndex | None = None
    _embeddings: np.ndarray | None = None
    _tfidf_vec: Any = None
    _tfidf_mat: Any = None
    _sbert_model: Any = field(default=None, repr=False)
    _corpus: list[str] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, store: ArtifactStore | None = None) -> Recommender:
        store = store or get_artifact_store()
        meta = json.loads(store.read_bytes(META_KEY).decode("utf-8"))
        backend = meta.get("backend", "tfidf")

        courses = pd.read_parquet(io.BytesIO(store.read_bytes(COURSES_KEY)))
        rec = cls(courses=courses, backend=backend, meta=meta, store=store)

        if backend == "sbert":
            rec._embeddings = np.load(io.BytesIO(store.read_bytes(EMB_KEY)))
            dim = int(meta.get("dim", rec._embeddings.shape[1]))
            if store.exists(INDEX_KEY):
                rec._index = VectorIndex.from_bytes(store.read_bytes(INDEX_KEY), dim=dim)
            else:  # rebuild from embeddings if the index artifact is absent
                rec._index = VectorIndex(dim=dim).add(rec._embeddings)
        else:
            from scipy import sparse

            rec._tfidf_vec = pickle.loads(store.read_bytes(TFIDF_VEC_KEY))
            rec._tfidf_mat = sparse.load_npz(io.BytesIO(store.read_bytes(TFIDF_MAT_KEY)))

        return rec

    # ------------------------------------------------------------- internals
    @property
    def corpus(self) -> list[str]:
        """Lower-cased searchable text per course (cached)."""
        if self._corpus is None:
            df = self.courses
            self._corpus = (
                df.get("title", pd.Series([""] * len(df))).fillna("").astype(str)
                + " | "
                + df.get("description", pd.Series([""] * len(df))).fillna("").astype(str)
                + " | "
                + df.get("skills", pd.Series([""] * len(df))).fillna("").astype(str)
            ).str.lower().tolist()
        return self._corpus

    def _encode_query(self, text: str) -> np.ndarray:
        if self.backend == "sbert":
            model = self._get_sbert()
            return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return self._tfidf_vec.transform([text])

    def _get_sbert(self):
        if self._sbert_model is None:
            from sentence_transformers import SentenceTransformer

            self._sbert_model = SentenceTransformer(self.meta["sbert_model"])
        return self._sbert_model

    def _raw_scores(self, query_vec) -> np.ndarray:
        """Similarity of ``query_vec`` against every course."""
        if self.backend == "sbert":
            scores, idx = self._index.search(query_vec, top_k=self.size)
            full = np.full(self.size, -1.0, dtype=np.float32)
            full[idx[0]] = scores[0]
            return full
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity(query_vec, self._tfidf_mat).ravel()

    @property
    def size(self) -> int:
        return len(self.courses)

    def _apply_filters(self, sims: np.ndarray, filters: dict[str, str] | None) -> np.ndarray:
        """Return a copy of ``sims`` with filtered-out rows set to ``-inf``."""
        if not filters:
            return sims
        sims = sims.copy()
        for col, value in filters.items():
            if not value or col not in self.courses.columns:
                continue
            col_vals = self.courses[col].astype(str).str.lower()
            mask = col_vals.str.contains(re.escape(str(value).lower()), na=False)
            sims[~mask.values] = -np.inf
        return sims

    def _row_to_reco(self, i: int, score: float) -> Recommendation:
        row = self.courses.iloc[i]
        return Recommendation(
            id=_opt(row.get("id", i)),
            title=str(row.get("title", "")),
            score=score,
            category=_opt(row.get("category")),
            level=_opt(row.get("level")),
            rating=_opt(row.get("rating")),
            url=_opt(row.get("url")),
            skills=_opt(row.get("skills")),
        )

    # ---------------------------------------------------------------- public
    def recommend(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
        use_rerank: bool = False,
        overfetch: int = 50,
    ) -> list[Recommendation]:
        """Return the top-k courses most relevant to a free-text ``query``."""
        qv = self._encode_query(query)
        sims = self._apply_filters(self._raw_scores(qv), filters)

        k0 = min(max(top_k, overfetch), self.size)
        cand = np.argpartition(-sims, kth=k0 - 1)[:k0]
        cand = cand[np.argsort(-sims[cand])]
        cand = [i for i in cand if np.isfinite(sims[i])]

        if use_rerank and cand:
            from .rerank import rerank

            docs = [self.corpus[i] for i in cand]
            ranked = rerank(query, docs, cand, [sims[i] for i in cand], top_k=top_k)
            return [self._row_to_reco(i, s) for i, s in ranked]

        return [self._row_to_reco(int(i), float(sims[i])) for i in cand[:top_k]]

    def similar_to(self, course_id: Any, top_k: int = 10) -> list[Recommendation]:
        """Return courses most similar to an existing course by its id."""
        matches = self.courses.index[self.courses["id"] == course_id].tolist()
        if not matches:
            # fall back to positional index if id lookup fails
            if isinstance(course_id, (int, np.integer)) and 0 <= course_id < self.size:
                pos = int(course_id)
            else:
                raise KeyError(f"Course id not found: {course_id!r}")
        else:
            pos = matches[0]

        if self.backend == "sbert":
            qv = self._embeddings[pos:pos + 1]
        else:
            qv = self._tfidf_mat[pos]
        sims = self._raw_scores(qv)
        sims[pos] = -np.inf
        k0 = min(top_k, self.size - 1)
        cand = np.argpartition(-sims, kth=k0 - 1)[:k0]
        cand = cand[np.argsort(-sims[cand])]
        return [self._row_to_reco(int(i), float(sims[i])) for i in cand[:top_k]]

    def get_course(self, course_id: Any) -> dict[str, Any] | None:
        matches = self.courses[self.courses["id"] == course_id]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()


def _opt(value: Any) -> Any:
    """Normalize NA/NaN to None and numpy scalars to native Python for JSON."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value
