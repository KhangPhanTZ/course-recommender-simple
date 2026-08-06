"""Build the artifact set baked into the self-contained demo image.

TF-IDF backend (no heavy models / torch / model downloads), so the demo image
stays small and boots fast on free-tier hosts.

Data source, in priority order:
    1. ``data/``            — real datasets you committed (Coursera + Udemy + edX);
                              every recognized file is merged onto one schema.
    2. ``examples/catalog/`` — the bundled synthetic multi-platform sample.

Both go through the same multi-platform ingestion (src/data/sources.py), so the
demo behaves exactly like a production build.
"""
import sys
import tempfile

import yaml

sys.path.insert(0, ".")

from src.data import discover, load_catalog  # noqa: E402
from src.pipeline import build  # noqa: E402
from src.storage import get_artifact_store  # noqa: E402

CONFIG = "config/demo.yaml"
DATA_DIR = "data"
SAMPLE_DIR = "examples/catalog"


def _has_datasets(data_dir: str) -> bool:
    """True when data/ holds at least one recognized course dataset."""
    try:
        return len(load_catalog(discover(data_dir))) > 0
    except ValueError:
        return False


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
    data_dir = DATA_DIR if _has_datasets(DATA_DIR) else SAMPLE_DIR
    catalog = load_catalog(discover(data_dir))
    n_rows = len(catalog)
    print(f"Building demo artifacts from: {data_dir}/ ({n_rows} courses)")

    store = get_artifact_store()
    build(_adaptive_config(CONFIG, n_rows), data_dir, store=store)
    print(f"Demo artifacts ready: {n_rows} courses.")


if __name__ == "__main__":
    main()
