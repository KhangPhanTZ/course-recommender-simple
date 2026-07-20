"""Cover the dense (SBERT-style) backend path without downloading a model.

We synthesize normalized embeddings + a serialized VectorIndex and load the
Recommender against them, exercising the FAISS/numpy index branch used by
``similar_to`` and ``_raw_scores``.
"""
import io
import json

import numpy as np
import pandas as pd

from src.recsys.index import VectorIndex
from src.recsys.recommender import Recommender
from src.storage import LocalArtifactStore


def _write_sbert_artifacts(store):
    # Four courses in 3-D; rows 0 and 1 are near-duplicates.
    emb = np.array(
        [[1.0, 0.0, 0.0],
         [0.95, 0.05, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    df = pd.DataFrame({
        "id": [0, 1, 2, 3],
        "title": ["Alpha", "Alpha Prime", "Beta", "Gamma"],
        "description": ["", "", "", ""],
        "skills": ["x", "x", "y", "z"],
        "category": ["c", "c", "d", "e"],
        "level": ["beginner", "beginner", "advanced", "advanced"],
        "rating": [4.0, 4.1, 4.2, 4.3],
        "url": [None, None, None, None],
    })

    buf = io.BytesIO()
    np.save(buf, emb)
    store.write_bytes("embeddings.npy", buf.getvalue())

    index = VectorIndex(dim=3, use_faiss=False).add(emb)
    store.write_bytes("vector.index", index.to_bytes())

    pbuf = io.BytesIO()
    df.to_parquet(pbuf, index=False)
    store.write_bytes("courses.parquet", pbuf.getvalue())

    meta = {"backend": "sbert", "dim": 3, "sbert_model": "unused-in-test", "n_rows": 4}
    store.write_bytes("meta.json", json.dumps(meta).encode())


def test_sbert_similar_to(tmp_path):
    store = LocalArtifactStore(base_dir=str(tmp_path))
    _write_sbert_artifacts(store)

    rec = Recommender.load(store)
    assert rec.backend == "sbert"

    hits = rec.similar_to(0, top_k=2)
    assert hits[0].id == 1          # nearest neighbour is the near-duplicate
    assert all(h.id != 0 for h in hits)  # excludes the query itself
