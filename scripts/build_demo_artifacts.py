"""Build the artifact set baked into the self-contained demo image.

TF-IDF backend (no heavy models / torch / model downloads), so the demo image
stays small and boots fast on free-tier hosts.
"""
import os
import sys
import tempfile

import yaml

sys.path.insert(0, ".")

from src.pipeline import build  # noqa: E402
from src.storage import get_artifact_store  # noqa: E402

CONFIG = "config/demo.yaml"
FULL_DATASET = "data/Coursera.csv"
SAMPLE_DATASET = "examples/sample_courses.csv"


def _adaptive_config(base_config: str, n_rows: int) -> str:
    """Write a temp config whose cluster count scales with the catalog size.

    Roughly one cluster per ~150 courses, clamped to 6..24, so clustering stays
    sensible for both the sample catalog and a full multi-thousand-course set.
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
    print(f"Demo artifacts ready: {n_rows} courses.")


if __name__ == "__main__":
    main()
