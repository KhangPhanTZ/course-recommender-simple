"""Tests for the fine-tuning pair builders (src/training/pairs.py).

These stay hermetic: no network, no LLM, no torch. The heavy training step in
src/training/finetune.py is exercised only on a GPU box / Colab, not in CI.
"""
from src.training.pairs import (
    build_llm_pairs,
    build_proxy_pairs,
    course_text,
    read_jsonl,
    write_jsonl,
)

_RECORDS = [
    {"id": 1, "title": "Deep Learning with PyTorch", "skills": "pytorch, neural networks",
     "description": "Build and train neural networks.", "category": "Specialization", "level": "intermediate"},
    {"id": 2, "title": "SQL for Data Analysis", "skills": "sql, databases",
     "description": "Query and aggregate data.", "category": "Course", "level": "beginner"},
    {"id": 3, "title": "", "skills": "", "description": "", "category": "", "level": ""},  # empty -> skipped
]


def test_course_text_joins_present_fields_and_skips_missing():
    text = course_text(_RECORDS[0])
    assert "Deep Learning with PyTorch" in text and "pytorch" in text
    assert course_text(_RECORDS[2]) == ""  # all fields empty


def test_build_proxy_pairs_are_wellformed_and_skip_empty():
    pairs = build_proxy_pairs(_RECORDS)
    assert pairs, "expected some proxy pairs"
    # the empty record (id 3) contributes nothing
    assert all(p["course_id"] != 3 for p in pairs)
    for p in pairs:
        assert p["query"] and p["positive"] and p["strategy"] == "proxy"
    # a course's skills should appear as one of its anchors
    assert any(p["query"] == "sql, databases" for p in pairs)


def test_build_llm_pairs_parses_lines_and_tolerates_failures():
    def fake_complete(system, prompt):
        # numbered + quoted lines exercise the parser's cleanup
        if "PyTorch" in prompt:
            return '1. "learn pytorch"\n2. deep learning course\n- neural nets training'
        raise RuntimeError("model unavailable")  # course 2 fails -> skipped, not fatal

    pairs = build_llm_pairs(_RECORDS, fake_complete, per_course=3)
    assert {p["course_id"] for p in pairs} == {1}  # only the first course produced queries
    queries = {p["query"] for p in pairs}
    assert "learn pytorch" in queries and "deep learning course" in queries
    assert all(p["strategy"] == "llm" for p in pairs)


def test_jsonl_roundtrip(tmp_path):
    pairs = build_proxy_pairs(_RECORDS)
    path = str(tmp_path / "pairs.jsonl")
    n = write_jsonl(pairs, path)
    assert n == len(pairs)
    assert read_jsonl(path) == pairs
