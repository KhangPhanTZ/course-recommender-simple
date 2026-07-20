"""Approximate nearest-neighbour (ANN) index over course embeddings.

Uses FAISS when available for fast inner-product search on normalized
vectors (equivalent to cosine similarity), and transparently falls back to a
vectorized numpy brute-force search when FAISS is not installed. This keeps
local dev and CI dependency-light while giving production the fast path.

Only dense float32 vectors are indexed. For TF-IDF (sparse) the pipeline
reduces dimensionality first, or the numpy fallback operates on the dense
matrix directly.
"""
from __future__ import annotations

import io

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class VectorIndex:
    """Cosine-similarity nearest-neighbour index.

    Parameters
    ----------
    dim:
        Embedding dimensionality.
    use_faiss:
        Force-enable/disable FAISS. ``None`` means "use it if importable".
    """

    def __init__(self, dim: int, use_faiss: bool | None = None) -> None:
        self.dim = dim
        self._faiss = None
        self._index = None          # faiss index, when available
        self._matrix: np.ndarray | None = None  # numpy fallback store
        self._use_faiss = self._resolve_faiss(use_faiss)

    @staticmethod
    def _resolve_faiss(use_faiss: bool | None):
        if use_faiss is False:
            return False
        try:
            import faiss  # noqa: F401
            return True
        except Exception:
            if use_faiss is True:
                raise
            return False

    # -- build -------------------------------------------------------------
    def add(self, vectors: np.ndarray) -> VectorIndex:
        """Add (and normalize) row-vectors to the index."""
        vecs = _l2_normalize(vectors)
        if self._use_faiss:
            import faiss

            self._faiss = faiss
            index = faiss.IndexFlatIP(self.dim)
            index.add(vecs)
            self._index = index
        else:
            self._matrix = vecs
        return self

    # -- query -------------------------------------------------------------
    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, indices)`` for the top-k most similar rows.

        ``query`` may be a single vector or a batch. Output shape is
        ``(n_queries, top_k)``.
        """
        q = _l2_normalize(np.atleast_2d(query))
        if self._use_faiss and self._index is not None:
            scores, idx = self._index.search(q, top_k)
            return scores, idx
        if self._matrix is None:
            raise RuntimeError("Index is empty; call add() before search().")
        sims = q @ self._matrix.T  # cosine because both sides normalized
        k = min(top_k, sims.shape[1])
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        # sort each row's top-k by score descending
        row = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[row, idx], axis=1)
        idx = idx[row, order]
        scores = sims[row, idx]
        return scores, idx

    @property
    def size(self) -> int:
        if self._use_faiss and self._index is not None:
            return self._index.ntotal
        return 0 if self._matrix is None else self._matrix.shape[0]

    # -- persistence -------------------------------------------------------
    def to_bytes(self) -> bytes:
        """Serialize the index to bytes for the artifact store."""
        if self._use_faiss and self._index is not None:
            import faiss

            data = faiss.serialize_index(self._index)
            return b"FAIS" + np.asarray(data, dtype=np.uint8).tobytes()
        buf = io.BytesIO()
        np.save(buf, self._matrix, allow_pickle=False)
        return b"NPY_" + buf.getvalue()

    @classmethod
    def from_bytes(cls, blob: bytes, dim: int, use_faiss: bool | None = None) -> VectorIndex:
        """Reconstruct a :class:`VectorIndex` from :meth:`to_bytes` output."""
        magic, payload = blob[:4], blob[4:]
        obj = cls(dim=dim, use_faiss=use_faiss)
        if magic == b"FAIS":
            import faiss

            arr = np.frombuffer(payload, dtype=np.uint8)
            obj._use_faiss = True
            obj._faiss = faiss
            obj._index = faiss.deserialize_index(arr)
            return obj
        if magic == b"NPY_":
            buf = io.BytesIO(payload)
            obj._use_faiss = False
            obj._matrix = np.load(buf, allow_pickle=False)
            return obj
        raise ValueError("Unrecognized VectorIndex serialization header.")
