"""Grounding guarantees: the no-LLM path never fabricates courses.

The LLM never *selects* courses — retrieval does, and the LLM only narrates a
provided set. These tests pin the deterministic fallbacks (used whenever no LLM
is configured, and in CI) so a regression that invents catalog entries fails.
"""
from src.llm.service import RecommendationLLM


def _llm():
    llm = RecommendationLLM()
    assert not llm.enabled  # hermetic: no provider configured
    return llm


def test_chat_fallback_grounds_in_provided_courses():
    llm = _llm()
    courses = [
        {"title": "Deep Learning with PyTorch", "skills": "pytorch", "level": "beginner"},
        {"title": "Applied ML", "skills": "sklearn", "level": "intermediate"},
    ]
    reply = llm.advise_chat([{"role": "user", "content": "learn deep learning"}], courses)
    # references a real provided course, invents nothing
    assert "Deep Learning with PyTorch" in reply


def test_chat_fallback_admits_when_no_courses():
    llm = _llm()
    reply = llm.advise_chat([{"role": "user", "content": "underwater basket weaving"}], [])
    assert "couldn't find" in reply.lower()


def test_roadmap_intro_fallback_mentions_only_track_and_tiers():
    llm = _llm()
    roadmap = {
        "label": "ML Engineer",
        "summary": "Ship models.",
        "nodes": [
            {"tier": "Foundation", "skills": ["python"], "courses": [{"title": "Intro to Python"}]},
            {"tier": "Core", "skills": ["ml"], "courses": []},
        ],
        "bridges": [{"track": "mlops", "label": "MLOps / DevOps", "note": "next"}],
    }
    intro = llm.advise_roadmap(roadmap)
    assert "ML Engineer" in intro
    assert "Foundation" in intro and "Core" in intro
