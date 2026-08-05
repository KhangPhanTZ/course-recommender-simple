"""Build the artifact set baked into the self-contained demo image.

TF-IDF backend (no heavy models), plus a 2-D TruncatedSVD projection written as
``umap_embedding.npy`` so the catalog Map view works without the ``umap-learn``
dependency. Real builds use UMAP (see ``src.pipeline``); this is a lightweight
stand-in for the demo only.
"""
import io
import sys

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, ".")

from src.pipeline import build  # noqa: E402
from src.storage import get_artifact_store  # noqa: E402

CONFIG = "config/demo.yaml"
DATA = "examples/sample_courses.csv"


def main() -> None:
    store = get_artifact_store()
    build(CONFIG, DATA, store=store)

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
