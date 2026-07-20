from src.recsys.recommender import Recommender
from src.storage import get_artifact_store


def _load(built_artifacts):
    return Recommender.load(get_artifact_store())


def test_recommend_relevance(built_artifacts):
    rec = _load(built_artifacts)
    hits = rec.recommend("deep learning with pytorch", top_k=3)
    assert hits, "expected at least one hit"
    assert "pytorch" in hits[0].title.lower() or "deep learning" in hits[0].title.lower()
    # scores are descending
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_recommend_with_level_filter(built_artifacts):
    rec = _load(built_artifacts)
    hits = rec.recommend("python", top_k=5, filters={"level": "beginner"})
    assert all((h.level or "").lower() == "beginner" for h in hits)


def test_similar_to(built_artifacts):
    rec = _load(built_artifacts)
    hits = rec.similar_to(0, top_k=3)
    assert len(hits) == 3
    assert all(h.id != 0 for h in hits)  # excludes the query course


def test_get_course(built_artifacts):
    rec = _load(built_artifacts)
    course = rec.get_course(0)
    assert course is not None
    assert "title" in course
