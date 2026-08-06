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

_ADVISOR_SYSTEM = (
    "You are a friendly learning advisor for an online-course catalog. Using "
    "ONLY the courses provided as context, help the learner understand what the "
    "relevant courses cover (topics, skills, level) and suggest a sensible "
    "development path — what to learn first and what to move on to next. Be "
    "concise (3-6 sentences), reference courses by title, and never invent "
    "courses that are not in the context. If the context is empty, say you "
    "couldn't find matching courses and ask the learner to describe a topic."
)

_ROADMAP_SYSTEM = (
    "You are a career-path advisor for an online-course catalog. Given a career "
    "track and its tiered roadmap (each tier lists skills and grounded courses), "
    "write a short, motivating overview of the path. Reference the tier names, "
    "stay concrete, and never invent courses beyond those provided."
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

    # ------------------------------------------------------------- chat advisor
    def advise_chat(self, history: list[dict[str, str]], courses: list[dict[str, Any]]) -> str:
        """Answer the learner's latest message, grounded in retrieved ``courses``."""
        if self.enabled:
            try:
                context = "\n".join(
                    f"- {c.get('title')} | skills: {c.get('skills') or 'n/a'} "
                    f"| level: {c.get('level') or 'n/a'}"
                    for c in courses[:8]
                )
                convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
                prompt = (
                    f"Conversation so far:\n{convo}\n\n"
                    f"Relevant courses:\n{context or '(none found)'}\n\n"
                    "Reply as the advisor to the learner's last message."
                )
                return self.provider.complete(
                    _ADVISOR_SYSTEM,
                    prompt,
                    max_tokens=self.settings.max_tokens,
                    temperature=self.settings.temperature,
                ).strip()
            except Exception:
                pass
        return self._template_chat(history, courses)

    # ---------------------------------------------------------- track roadmap
    def advise_roadmap(self, roadmap: dict[str, Any]) -> str:
        """A short intro for a career-track roadmap, grounded in its tiers/courses."""
        label = roadmap.get("label", "this track")
        tiers = roadmap.get("nodes", [])
        if self.enabled:
            try:
                context = "\n".join(
                    f"- {n['tier']}: skills {', '.join(n.get('skills', []))}; "
                    f"courses: {', '.join(str(c.get('title')) for c in n.get('courses', [])) or 'n/a'}"
                    for n in tiers
                )
                prompt = (
                    f"Career track: {label}\n\nRoadmap tiers (grounded in the catalog):\n{context}\n\n"
                    "Write a short, motivating 2-3 sentence overview of this path for a "
                    "learner: what they'll build tier by tier and where it leads. Reference "
                    "the tier names. Do not invent courses beyond those listed."
                )
                return self.provider.complete(
                    _ROADMAP_SYSTEM, prompt,
                    max_tokens=self.settings.max_tokens, temperature=self.settings.temperature,
                ).strip()
            except Exception:
                pass
        return self._template_roadmap(roadmap)

    @staticmethod
    def _template_roadmap(roadmap: dict[str, Any]) -> str:
        label = roadmap.get("label", "this track")
        tiers = " → ".join(n["tier"] for n in roadmap.get("nodes", []))
        bridges = roadmap.get("bridges", [])
        nxt = ""
        if bridges:
            nxt = " From there you can branch into " + ", ".join(b["label"] for b in bridges) + "."
        return (
            f"Here's a {label} roadmap grounded in the catalog: {tiers}. "
            f"{roadmap.get('summary', '')} Work tier by tier, taking the linked courses "
            f"at each stage before moving on.{nxt}"
        )

    @staticmethod
    def _template_chat(history: list[dict[str, str]], courses: list[dict[str, Any]]) -> str:
        last = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        if not courses:
            return (
                "I couldn't find matching courses for that. Try describing a topic "
                "or skill you'd like to learn (e.g. \"data analysis with Python\")."
            )
        top = courses[0]
        names = ", ".join(str(c.get("title")) for c in courses[:4])
        skills = top.get("skills") or "the core skills"
        return (
            f"For “{last}”, a good starting point is “{top.get('title')}” "
            f"(covers {skills}; level: {top.get('level') or 'all levels'}). "
            f"Related options: {names}. Begin with a beginner course to build "
            "fundamentals, then move to intermediate/advanced ones that go deeper "
            "into the same skills."
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
