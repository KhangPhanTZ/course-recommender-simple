from src.llm.provider import (
    AnthropicProvider,
    BedrockProvider,
    NullProvider,
    get_provider,
)
from src.llm.service import RecommendationLLM
from src.settings import LLMSettings


def test_null_provider_unavailable():
    p = NullProvider()
    assert p.available is False


def test_get_provider_disabled():
    s = LLMSettings(provider="disabled", enabled=True)
    assert isinstance(get_provider(s), NullProvider)


def test_get_provider_selection():
    assert isinstance(get_provider(LLMSettings(provider="anthropic")), AnthropicProvider)
    assert isinstance(get_provider(LLMSettings(provider="bedrock")), BedrockProvider)


def test_heuristic_query_understanding():
    llm = RecommendationLLM(provider=NullProvider())
    out = llm.understand_query("I want to learn deep learning for beginners")
    assert out["level"] == "beginner"
    assert "deep learning" in out["search"].lower()


def test_template_explanation_fallback():
    llm = RecommendationLLM(provider=NullProvider())
    courses = [{"title": "Deep Learning", "skills": "pytorch", "level": "beginner"}]
    text = llm.explain_recommendations("learn dl", courses)
    assert "Deep Learning" in text


def test_explain_empty():
    llm = RecommendationLLM(provider=NullProvider())
    assert "No matching" in llm.explain_recommendations("x", [])


class _StubProvider:
    available = True

    def complete(self, system, prompt, *, max_tokens=512, temperature=0.2):
        if "query-understanding" in system:
            return '{"search": "pytorch deep learning", "level": "beginner", "category": null}'
        return "These courses fit your goal."


def test_llm_query_understanding_parsing():
    llm = RecommendationLLM(provider=_StubProvider())
    out = llm.understand_query("messy request about pytorch")
    assert out["search"] == "pytorch deep learning"
    assert out["level"] == "beginner"
    assert out["category"] is None


def test_llm_explanation_uses_provider():
    llm = RecommendationLLM(provider=_StubProvider())
    text = llm.explain_recommendations("goal", [{"title": "X", "skills": "y", "level": "z"}])
    assert text == "These courses fit your goal."
