"""Tests for offline evaluation (src/evaluation.py + /metrics endpoint)."""
import json

from fastapi.testclient import TestClient

from src.api import deps
from src.api.main import app
from src.evaluation import _ndcg, _percentile, evaluate
from src.recsys.recommender import Recommender
from src.storage import get_artifact_store


def _client(built_artifacts):
    deps.reset_state()
    return TestClient(app)


def test_ndcg_and_percentile_helpers():
    assert _ndcg([1, 1, 1], 3) == 1.0          # perfectly ranked
    assert _ndcg([0, 0, 0], 3) == 0.0          # nothing relevant
    assert _ndcg([1, 0, 0], 3) > _ndcg([0, 0, 1], 3)  # earlier hit ranks higher
    assert _percentile([10, 20, 30, 40], 50) in (20, 30)
    assert _percentile([], 95) == 0.0


def test_evaluate_returns_wellformed_metrics(built_artifacts):
    deps.reset_state()
    rec = Recommender.load(get_artifact_store())
    m = evaluate(rec, k=5, sample=50)

    assert m["backend"] == "tfidf"
    assert m["n_courses"] == rec.size
    assert m["k"] == 5
    # latency is always measured
    assert set(m["latency_ms"]) == {"p50", "p95", "mean"}
    assert m["latency_ms"]["p95"] >= 0
    # retrieval metrics are present and in range when categories exist
    for key in ("precision_at_k", "recall_at_k", "hit_rate", "mrr", "ndcg_at_k"):
        assert 0.0 <= m[key] <= 1.0


def test_metrics_endpoint_absent(built_artifacts):
    c = _client(built_artifacts)
    body = c.get("/metrics").json()
    assert body["available"] is False


def test_metrics_endpoint_present(built_artifacts):
    # Write a metrics.json into the same store the API reads from.
    store = get_artifact_store()
    store.write_bytes("metrics.json", json.dumps({"k": 10, "ndcg_at_k": 0.83}).encode())
    c = _client(built_artifacts)
    body = c.get("/metrics").json()
    assert body["available"] is True
    assert body["ndcg_at_k"] == 0.83
