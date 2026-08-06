"""Tests for multi-platform dataset ingestion (src/data/sources.py)."""
import pandas as pd
import pytest

from src.data import (
    detect_spec,
    load_catalog,
    normalize_level,
    normalize_source,
    source_counts,
)

COURSERA = pd.DataFrame({
    "course": ["Machine Learning", "Deep Learning"],
    "partner": ["Stanford", "DeepLearning.AI"],
    "skills": ["python, regression", "pytorch, cnn"],
    "certificatetype": ["Course", "Specialization"],
    "level": ["Beginner", "Advanced"],
    "rating": [4.8, 4.9],
})

UDEMY = pd.DataFrame({
    "course_id": [1, 2],
    "course_title": ["Complete SQL Bootcamp", "React for QA Automation"],
    "url": ["http://u/1", "http://u/2"],
    "is_paid": [True, True], "price": [50, 40],
    "num_subscribers": [1000, 500], "num_reviews": [10, 5],
    "num_lectures": [20, 15], "level": ["All Levels", "Intermediate Level"],
    "content_duration": [3.5, 2.0], "published_timestamp": ["2020", "2021"],
    "subject": ["Development", "Development"],
})

EDX = pd.DataFrame({
    "title": ["Product Management Essentials"],
    "summary": ["intro"], "n_enrolled": [1000], "course_type": ["self"],
    "institution": ["HarvardX"], "instructors": ["Staff"], "Level": ["Introductory"],
    "subject": ["Business & Management"], "language": ["English"], "subtitles": ["en"],
    "course_effort": ["3h"], "course_length": ["6w"], "price": ["Free"],
    "course_description": ["Learn PM fundamentals: roadmaps, stakeholders, metrics."],
    "course_syllabus": ["Week 1: discovery. Week 2: roadmap."],
    "course_url": ["http://edx/pm"],
})


@pytest.mark.parametrize("df,expected", [(COURSERA, "coursera"), (UDEMY, "udemy"), (EDX, "edx")])
def test_detect_spec(df, expected):
    spec = detect_spec(df)
    assert spec is not None and spec.name == expected


def test_detect_spec_unknown():
    assert detect_spec(pd.DataFrame({"foo": [1], "bar": [2]})) is None


def test_normalize_level():
    assert normalize_level("Beginner Level") == "beginner"
    assert normalize_level("Introductory") == "beginner"
    assert normalize_level("Expert Level") == "advanced"
    assert normalize_level("All Levels") is None   # no usable signal
    assert normalize_level(None) is None
    assert normalize_level(float("nan")) is None


def test_normalize_source_edx_keeps_syllabus_and_provider():
    out = normalize_source(EDX, detect_spec(EDX))
    row = out.iloc[0]
    assert row["source"] == "edx"
    assert row["provider"] == "HarvardX"          # from institution column
    assert row["level"] == "beginner"             # Introductory -> beginner
    assert "roadmap" in row["syllabus"].lower()   # syllabus preserved for RAG
    assert row["category"] == "Business & Management"


def test_normalize_source_udemy_constant_provider():
    out = normalize_source(UDEMY, detect_spec(UDEMY))
    assert (out["provider"] == "Udemy").all()
    assert (out["source"] == "udemy").all()
    assert pd.isna(out.iloc[0]["level"])          # "All Levels" -> unset


def test_load_catalog_merges_and_assigns_ids(tmp_path):
    paths = []
    for name, df in [("coursera", COURSERA), ("udemy", UDEMY), ("edx", EDX)]:
        p = tmp_path / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(str(p))

    catalog = load_catalog(paths)
    assert len(catalog) == 5                       # 2 + 2 + 1
    assert list(catalog["id"]) == [0, 1, 2, 3, 4]  # contiguous, stable
    assert source_counts(catalog) == {"coursera": 2, "udemy": 2, "edx": 1}
    # canonical schema present
    for col in ("title", "provider", "level", "source", "syllabus"):
        assert col in catalog.columns


def test_load_catalog_dedups_title_provider(tmp_path):
    dup = pd.concat([COURSERA, COURSERA.iloc[[0]]], ignore_index=True)  # repeat one row
    p = tmp_path / "coursera.csv"
    dup.to_csv(p, index=False)
    catalog = load_catalog([str(p)])
    assert len(catalog) == 2                        # exact (title, provider) repeat dropped


def test_load_catalog_skips_unknown_but_raises_when_empty(tmp_path):
    junk = tmp_path / "junk.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(junk, index=False)
    with pytest.raises(ValueError):
        load_catalog([str(junk)])


def test_pipeline_builds_from_directory(tmp_path, monkeypatch):
    """Building against a directory merges all sources and records provenance."""
    import yaml

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    COURSERA.to_csv(data_dir / "coursera.csv", index=False)
    UDEMY.to_csv(data_dir / "udemy.csv", index=False)
    EDX.to_csv(data_dir / "edx.csv", index=False)

    cfg = {
        "columns": {"title": "title", "description": "description"},  # ignored for dirs
        "text_fields": ["title", "skills", "description", "syllabus", "category", "level"],
        "min_characters": 3,
        "use_sbert": False,
        "kmeans_k": 2,
        "top_k": 5,
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setenv("ARTIFACT_STORE", "local")
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("LLM_PROVIDER", "disabled")

    from src.api import deps
    from src.pipeline import build
    from src.recsys.recommender import Recommender
    from src.storage import get_artifact_store

    build(str(cfg_path), str(data_dir), store=get_artifact_store())
    deps.reset_state()

    rec = Recommender.load(get_artifact_store())
    assert rec.meta["sources"] == {"coursera": 2, "udemy": 2, "edx": 1}

    hits = rec.recommend("product management roadmap", top_k=3)
    assert hits and hits[0].to_dict()["source"] in {"coursera", "udemy", "edx"}
    assert any(h.to_dict()["provider"] for h in hits)
