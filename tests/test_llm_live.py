"""Live tests against a real LLM provider.

Skipped unless RUN_LLM_TESTS=1, because they cost money, need network access,
and assert on model output rather than deterministic code. Keep them out of the
default suite and out of CI; run them when changing prompts or upgrading models:

    RUN_LLM_TESTS=1 pytest tests/test_llm_live.py -v
"""
from __future__ import annotations

import os

import pytest

from src.llm.provider import get_provider
from src.llm.service import RecommendationLLM
from src.settings import get_settings

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LLM_TESTS") != "1",
    reason="set RUN_LLM_TESTS=1 to run live LLM tests (costs tokens)",
)

TEMPLATE_PREFIX = "Based on your goal"
VIETNAMESE_QUERY = "toi muon hoc deep learning voi pytorch, trinh do moi bat dau"

COURSES = [
    {"title": "Deep Learning with PyTorch", "level": "Beginner",
     "skills": "Neural Networks, PyTorch"},
    {"title": "Financial Modeling", "level": "Advanced",
     "skills": "Valuation, Cash Flow Analysis"},
]


@pytest.fixture(scope="module")
def service() -> RecommendationLLM:
    """A service bound to whatever provider .env configures."""
    settings = get_settings()
    if not settings.llm.is_active:
        pytest.skip("LLM_PROVIDER is disabled")
    return RecommendationLLM(get_provider(settings.llm), settings.llm)


def test_provider_round_trip(service: RecommendationLLM) -> None:
    """Credentials and transport work. Isolates auth from prompt behaviour.

    max_tokens must leave room for reasoning as well as the answer: recent
    models think by default and the budget covers both, so a very small value
    yields an empty text block rather than a short reply.
    """
    reply = service.provider.complete(
        "Answer with a single word.", "Reply OK if you receive this.", max_tokens=256
    )
    assert "OK" in reply.upper()


def test_understands_non_english_query(service: RecommendationLLM) -> None:
    """The heuristic fallback cannot translate, so a rewrite proves the model ran."""
    parsed = service.understand_query(VIETNAMESE_QUERY)

    assert parsed["search"].strip() != VIETNAMESE_QUERY.strip()
    assert parsed["level"] == "beginner"


def test_explanation_is_generated_not_templated(service: RecommendationLLM) -> None:
    """Guards the silent-degradation path: a template here means the LLM failed."""
    explanation = service.explain_recommendations(VIETNAMESE_QUERY, COURSES)

    assert explanation
    assert not explanation.startswith(TEMPLATE_PREFIX)


def test_explanation_is_grounded_in_the_retrieved_courses(
    service: RecommendationLLM,
) -> None:
    """The explanation must discuss what was retrieved, not invent a syllabus."""
    explanation = service.explain_recommendations(VIETNAMESE_QUERY, COURSES)

    assert "PyTorch" in explanation
