"""Build the artifact set baked into the self-contained demo image.

TF-IDF backend (no heavy models), plus a 2-D TruncatedSVD projection written as
``umap_embedding.npy`` so the catalog Map view works without the ``umap-learn``
dependency. Real builds use UMAP (see ``src.pipeline``); this is a lightweight
stand-in for the demo only.
"""
import io
import os
import sys
import tempfile

import numpy as np
import yaml
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, ".")

from src.pipeline import build  # noqa: E402
from src.storage import get_artifact_store  # noqa: E402

CONFIG = "config/demo.yaml"
FULL_DATASET = "data/Coursera.csv"
SAMPLE_DATASET = "examples/sample_courses.csv"


def _adaptive_config(base_config: str, n_rows: int) -> str:
    """Write a temp config whose cluster count scales with the catalog size.

    Keeps the UMAP map readable for both the 128-course sample and a
    full multi-thousand-course dataset (roughly one cluster per ~150 courses,
    clamped to 6..24).
    """
    cfg = yaml.safe_load(open(base_config))
    cfg["kmeans_k"] = max(6, min(24, n_rows // 150))
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name


def main() -> None:
    # Use the full Coursera dataset when it's committed, else the sample.
    data = FULL_DATASET if os.path.isfile(FULL_DATASET) else SAMPLE_DATASET
    with open(data, encoding="utf-8") as f:
        n_rows = max(0, sum(1 for _ in f) - 1)
    print(f"Building demo artifacts from: {data} ({n_rows} rows)")

    store = get_artifact_store()
    build(_adaptive_config(CONFIG, n_rows), data, store=store)

    # 2-D projection for the Map view (SVD stands in for UMAP in the lite image).
    X = sparse.load_npz(io.BytesIO(store.read_bytes("X_tfidf.npz")))
    n_comp = min(2, X.shape[1] - 1) if X.shape[1] > 1 else 1
    emb = TruncatedSVD(n_components=max(n_comp, 2), random_state=42).fit_transform(X)
    emb = np.asarray(emb[:, :2], dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, emb)
    store.write_bytes("umap_embedding.npy", buf.getvalue())
    print(f"Demo artifacts ready: {X.shape[0]} courses, 2-D map written.")


if __name__ == "__main__":
    main()
