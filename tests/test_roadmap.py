"""Tests for career-track roadmaps (src/llm/tracks.py + /roadmap endpoints)."""
from fastapi.testclient import TestClient

from src.api import deps
from src.api.main import app
from src.llm.tracks import (
    GROUP_ORDER,
    TIER_ORDER,
    TRACKS,
    assemble_roadmap,
    covered_tracks,
    get_track,
    list_tracks,
)


def _client(built_artifacts):
    deps.reset_state()
    return TestClient(app)


def _fake_retrieve(query, k):
    return [
        {"id": f"{query[:4]}-{i}", "title": f"Course {i}", "score": 1.0 - i * 0.1,
         "level": "beginner", "skills": query, "source": "coursera", "provider": "X"}
        for i in range(k)
    ]


def test_registry_is_wellformed():
    assert len(TRACKS) >= 15  # expanded beyond the original six
    for tid, track in TRACKS.items():
        assert track["tiers"], f"{tid} has no tiers"
        assert [t["name"] for t in track["tiers"]] == TIER_ORDER
        assert track["group"] in GROUP_ORDER, f"{tid} has unknown group {track['group']}"
        assert all(t["skills"] for t in track["tiers"]), f"{tid} has an empty tier"
        for bid in track.get("bridges", []):
            assert bid in TRACKS, f"{tid} bridges to unknown track {bid}"


def test_covered_tracks_gate():
    # a retriever that always returns courses -> every track qualifies
    assert len(covered_tracks(_fake_retrieve)) == len(TRACKS)
    # a retriever that returns nothing -> fall back to the full menu
    assert len(covered_tracks(lambda q, k: [])) == len(TRACKS)


def test_assemble_roadmap_structure_and_dedup():
    rm = assemble_roadmap("ml-engineer", _fake_retrieve, per_tier=2)
    assert rm["label"] == "ML Engineer"
    assert [n["tier"] for n in rm["nodes"]] == TIER_ORDER
    # progress edges chain the tiers
    assert [(e["source"], e["target"]) for e in rm["edges"]] == [
        ("foundation", "core"), ("core", "advanced"), ("advanced", "specialization")
    ]
    # every course appears at most once across tiers
    ids = [c["id"] for n in rm["nodes"] for c in n["courses"]]
    assert len(ids) == len(set(ids))
    assert rm["bridges"] and rm["bridges"][0]["track"] in TRACKS


def test_assemble_roadmap_unknown_track():
    assert assemble_roadmap("does-not-exist", _fake_retrieve) is None


def test_list_and_get_track():
    ids = {t["id"] for t in list_tracks()}
    assert {"data-analyst", "ml-engineer", "qa-engineer"} <= ids
    assert get_track("ml-engineer")["label"] == "ML Engineer"
    assert get_track("nope") is None


def test_roadmap_tracks_endpoint(built_artifacts):
    c = _client(built_artifacts)
    r = c.get("/roadmap/tracks")
    assert r.status_code == 200
    body = r.json()
    assert any(t["id"] == "ml-engineer" for t in body)
    assert all({"id", "label", "summary", "group"} <= t.keys() for t in body)
    assert all(t["group"] in GROUP_ORDER for t in body)


def test_roadmap_endpoint_builds_grounded_roadmap(built_artifacts):
    c = _client(built_artifacts)
    r = c.post("/roadmap", json={"track": "ml-engineer", "per_tier": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "ML Engineer"
    assert len(body["nodes"]) == 4
    assert body["intro"]  # deterministic fallback always returns text
    # at least one tier should surface a real catalog course (fixture has ML/DL courses)
    assert sum(len(n["courses"]) for n in body["nodes"]) > 0


def test_roadmap_endpoint_unknown_track(built_artifacts):
    c = _client(built_artifacts)
    assert c.post("/roadmap", json={"track": "astronaut"}).status_code == 404
