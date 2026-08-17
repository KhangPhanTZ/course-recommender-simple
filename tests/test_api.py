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


def test_courses_catalog_lists_and_paginates(built_artifacts):
    c = _client(built_artifacts)
    body = c.get("/courses", params={"limit": 3}).json()
    assert body["total"] == 8            # fixture catalog size
    assert len(body["items"]) == 3       # page honored
    assert {"id", "title", "skills", "description"} <= body["items"][0].keys()
    # offset advances the window
    page2 = c.get("/courses", params={"limit": 3, "offset": 3}).json()
    assert page2["items"][0]["id"] != body["items"][0]["id"]


def test_courses_have_resolved_urls(built_artifacts):
    c = _client(built_artifacts)
    # /recommend hits carry a real url + a url_direct flag
    hit = c.post("/recommend", json={"query": "deep learning", "top_k": 1}).json()["results"][0]
    assert hit["url"] and hit["url"].startswith("http")
    assert isinstance(hit["url_direct"], bool)
    # the fixture has no url column -> Coursera search deep-link, not direct
    assert "coursera.org/search?query=" in hit["url"]
    assert hit["url_direct"] is False
    # catalog + single-course endpoints resolve too
    item = c.get("/courses", params={"limit": 1}).json()["items"][0]
    assert item["url"].startswith("http") and item["url_direct"] is False
    detail = c.get(f"/courses/{item['id']}").json()
    assert detail["url"].startswith("http") and detail["url_direct"] is False


def test_courses_catalog_filters(built_artifacts):
    c = _client(built_artifacts)
    # the fixture has a "beginner" level and pytorch/deep-learning skills
    lvl = c.get("/courses", params={"level": "beginner"}).json()
    assert lvl["total"] >= 1
    assert all((it["level"] or "").lower() == "beginner" for it in lvl["items"])
    q = c.get("/courses", params={"q": "pytorch"}).json()
    assert q["total"] >= 1


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


def test_conversation_query_blends_recent_user_turns():
    from src.api.routes import _conversation_query
    from src.api.schemas import ChatMessage

    msgs = [
        ChatMessage(role="user", content="deep learning with pytorch"),
        ChatMessage(role="assistant", content="Here are some courses."),
        ChatMessage(role="user", content="what should I learn next?"),
    ]
    q = _conversation_query(msgs)
    # the short follow-up alone has no topic; the blended query keeps the thread
    assert "next" in q and "pytorch" in q
    # the latest message anchors the query (listed first)
    assert q.startswith("what should I learn next?")


def test_chat_followup_stays_grounded_in_context(built_artifacts):
    c = _client(built_artifacts)
    r = c.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "deep learning with pytorch"},
                {"role": "assistant", "content": "Here are some courses."},
                {"role": "user", "content": "what next?"},
            ]
        },
    )
    assert r.status_code == 200
    # a bare "what next?" would retrieve nothing on its own; context keeps it grounded
    assert len(r.json()["courses"]) > 0


def test_root(built_artifacts):
    c = _client(built_artifacts)
    assert c.get("/").json()["service"] == "course-recommender"
