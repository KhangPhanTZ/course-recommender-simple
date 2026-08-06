"""Evaluate the built artifacts and persist a metrics.json alongside them.

Reads the artifacts through the same ArtifactStore the API uses (local or S3),
so it works identically on a laptop and in CI/AWS:

    python scripts/run_eval.py
    ARTIFACT_STORE=s3 ARTIFACT_S3_BUCKET=... python scripts/run_eval.py
"""
import json
import sys

sys.path.insert(0, ".")

from src.evaluation import evaluate  # noqa: E402
from src.recsys.recommender import Recommender  # noqa: E402
from src.storage import get_artifact_store  # noqa: E402


def main() -> None:
    store = get_artifact_store()
    rec = Recommender.load(store)
    metrics = evaluate(rec)
    store.write_bytes("metrics.json", json.dumps(metrics, indent=2).encode("utf-8"))
    print("Wrote metrics.json:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
