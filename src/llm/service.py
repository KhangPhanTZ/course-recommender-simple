"""GenAI / RAG features layered on top of the retrieval engine.

Two capabilities, each with a deterministic fallback so the API always works
even without an LLM configured:

1. ``understand_query`` — turns a messy natural-language request into a clean
   search string plus structured filters (level, category). This is a light
   query-rewriting / intent-extraction step.
2. ``explain_recommendations`` — a Retrieval-Augmented Generation step: the
   retrieved courses are the grounding context, and the LLM writes a short,
   per-user rationale for *why* the set fits the request.

The provider is injected, so the same service runs against the Claude API or
AWS Bedrock unchanged.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..settings import LLMSettings
from .provider import LLMProvider, get_provider

_KNOWN_LEVELS = ["beginner", "intermediate", "advanced"]

_UNDERSTAND_SYSTEM = (
    "You are a query-understanding module for a course search engine. "
    "Given a user's natural-language learning goal, extract a concise semantic "
    "search query and optional filters. Respond with ONLY a JSON object of the "
    'form {"search": str, "level": str|null, "category": str|null}. '
    "Use one of beginner/intermediate/advanced for level when clearly implied, "
    "otherwise null. Do not add commentary."
)

_EXPLAIN_SYSTEM = (
    "You are a helpful learning advisor. Given a learner's goal and a list of "
    "recommended courses (title, skills, level), write a brief, friendly "
    "explanation (2-4 sentences) of why these courses suit the goal and how to "
    "sequence them. Be specific and reference course titles. Do not invent "
    "courses beyond the provided list."
)


class RecommendationLLM:
    """Bundles the GenAI features and their non-LLM fallbacks."""

    def __init__(self, provider: LLMProvider | None = None, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings()
        self.provider = provider or get_provider(self.settings)

    @property
    def enabled(self) -> bool:
        return self.provider.available

    # ----------------------------------------------------- query understanding
    def understand_query(self, text: str) -> dict[str, Any]:
        """Return ``{"search": str, "level": str|None, "category": str|None}``."""
        if self.enabled:
            try:
                raw = self.provider.complete(
                    _UNDERSTAND_SYSTEM,
                    f"User request: {text}",
                    max_tokens=200,
                    temperature=0.0,
                )
                parsed = _extract_json(raw)
                if parsed and parsed.get("search"):
                    return {
                        "search": str(parsed.get("search") or text),
                        "level": _clean(parsed.get("level")),
                        "category": _clean(parsed.get("category")),
                    }
            except Exception:
                pass  # fall through to heuristic
        return self._heuristic_understand(text)

    @staticmethod
    def _heuristic_understand(text: str) -> dict[str, Any]:
        low = text.lower()
        level = next((lv for lv in _KNOWN_LEVELS if lv in low), None)
        if level is None and re.search(r"\bfor beginners?\b|\bnewbie\b|\bfrom scratch\b", low):
            level = "beginner"
        # strip the level phrase from the semantic query to reduce noise
        search = re.sub(r"\bfor beginners?\b", "", text, flags=re.I).strip() or text
        return {"search": search, "level": level, "category": None}

    # ----------------------------------------------------------- explanations
    def explain_recommendations(self, goal: str, courses: list[dict[str, Any]]) -> str:
        """RAG: ground an explanation in the retrieved ``courses``."""
        if not courses:
            return "No matching courses were found for this request."
        if self.enabled:
            try:
                context = "\n".join(
                    f"- {c.get('title')} | skills: {c.get('skills') or 'n/a'} | level: {c.get('level') or 'n/a'}"
                    for c in courses[:8]
                )
                return self.provider.complete(
                    _EXPLAIN_SYSTEM,
                    f"Learner goal: {goal}\n\nRecommended courses:\n{context}",
                    max_tokens=self.settings.max_tokens,
                    temperature=self.settings.temperature,
                ).strip()
            except Exception:
                pass
        return self._template_explanation(goal, courses)

    @staticmethod
    def _template_explanation(goal: str, courses: list[dict[str, Any]]) -> str:
        top = courses[0]
        names = ", ".join(str(c.get("title")) for c in courses[:3])
        return (
            f"Based on your goal “{goal}”, the strongest match is "
            f"“{top.get('title')}”. Related picks include {names}. "
            "Start with the highest-scored course, then branch into the others "
            "to broaden the skills they share."
        )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object embedded in an LLM response."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None
