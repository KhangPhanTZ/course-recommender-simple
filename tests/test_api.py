from fastapi.testclient import TestClient

from src.api import deps
from src.api.main import app


def _client(built_artifacts):
    deps.reset_state()  # force reload against the temp artifacts
    return TestClient(app)


def test_health_ok(built_artifacts):
    c = _client(built_artifacts)
    body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "tfidf"
    assert body["n_courses"] == 8


def test_recommend_endpoint(built_artifacts):
    c = _client(built_artifacts)
    r = c.post("/recommend", json={"query": "deep learning with pytorch", "top_k": 3, "explain": True})
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 3
    assert data["explanation"]  # template fallback always returns text
    assert data["results"][0]["score"] >= data["results"][1]["score"]


def test_recommend_understands_level(built_artifacts):
    c = _client(built_artifacts)
    r = c.post("/recommend", json={"query": "python for beginners", "top_k": 5})
    data = r.json()
    assert data["filters"].get("level") == "beginner"
    assert all((h["level"] or "").lower() == "beginner" for h in data["results"])


def test_similar_endpoint(built_artifacts):
    c = _client(built_artifacts)
    r = c.post("/similar", json={"course_id": 0, "top_k": 3})
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_course_not_found(built_artifacts):
    c = _client(built_artifacts)
    assert c.get("/courses/9999").status_code == 404


def test_chat_endpoint(built_artifacts):
    c = _client(built_artifacts)
    r = c.post("/chat", json={"messages": [{"role": "user", "content": "deep learning with pytorch"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"]                    # template fallback always returns text
    assert len(body["courses"]) > 0         # grounded in retrieved courses


def test_chat_requires_messages(built_artifacts):
    c = _client(built_artifacts)
    assert c.post("/chat", json={"messages": []}).status_code == 422


def test_map_endpoint_graceful_without_viz(built_artifacts):
    # The test fixture builds with compute_viz disabled -> no UMAP artifact.
    c = _client(built_artifacts)
    body = c.get("/map").json()
    assert body["available"] is False
    assert body["points"] == []


def test_root(built_artifacts):
    c = _client(built_artifacts)
    assert c.get("/").json()["service"] == "course-recommender"
