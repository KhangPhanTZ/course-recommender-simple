import numpy as np

from src.recsys.index import VectorIndex


def _toy_vectors():
    return np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )


def test_numpy_fallback_search():
    idx = VectorIndex(dim=2, use_faiss=False).add(_toy_vectors())
    scores, ids = idx.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=2)
    assert ids.shape == (1, 2)
    # nearest to [1,0] is row 0, then row 1
    assert ids[0, 0] == 0
    assert ids[0, 1] == 1
    assert scores[0, 0] >= scores[0, 1]


def test_index_size():
    idx = VectorIndex(dim=2, use_faiss=False).add(_toy_vectors())
    assert idx.size == 4


def test_serialization_roundtrip_numpy():
    idx = VectorIndex(dim=2, use_faiss=False).add(_toy_vectors())
    blob = idx.to_bytes()
    restored = VectorIndex.from_bytes(blob, dim=2, use_faiss=False)
    s1, i1 = idx.search(np.array([[0.0, 1.0]], dtype=np.float32), top_k=1)
    s2, i2 = restored.search(np.array([[0.0, 1.0]], dtype=np.float32), top_k=1)
    assert i1[0, 0] == i2[0, 0] == 2


def test_search_before_add_raises():
    idx = VectorIndex(dim=2, use_faiss=False)
    try:
        idx.search(np.zeros((1, 2), dtype=np.float32), top_k=1)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
