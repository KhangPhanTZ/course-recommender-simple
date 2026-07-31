#!/usr/bin/env python
"""Manual smoke test for the GenAI layer.

Exercises the live LLM provider end to end and reports, for each stage,
whether the answer came from the model or from the deterministic fallback.
The distinction matters: the service degrades silently, so a working-looking
response is not evidence that the provider is reachable.

    python scripts/smoke_llm.py
    python scripts/smoke_llm.py --query "khoa hoc SQL cho nguoi moi"

Exit code is 0 when the model answered every stage, 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow `python scripts/smoke_llm.py` as well as `python -m scripts.smoke_llm`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.provider import get_provider
from src.llm.service import RecommendationLLM
from src.settings import get_settings

COURSES = [
    {"title": "Deep Learning with PyTorch", "level": "Beginner",
     "skills": "Neural Networks, PyTorch, Backpropagation"},
    {"title": "Machine Learning Foundations", "level": "Beginner",
     "skills": "Regression, Classification, Scikit-Learn"},
]


def mask(secret: str | None) -> str:
    if not secret:
        return "(not set)"
    return f"{secret[:14]}...{secret[-4:]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="toi muon hoc deep learning voi pytorch, trinh do moi bat dau",
        help="Natural-language goal to send through the pipeline.",
    )
    args = parser.parse_args()

    settings = get_settings()
    print("=" * 62)
    print(f"provider : {settings.llm.provider}")
    print(f"model    : {settings.llm.model}")
    print(f"api key  : {mask(os.getenv('ANTHROPIC_API_KEY'))}")
    print(f"enabled  : {settings.llm.enabled}")
    print("=" * 62)

    if not settings.llm.is_active:
        print("\nLLM_PROVIDER is 'disabled' — nothing to smoke test.")
        print("Set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY in .env.")
        return 1

    provider = get_provider(settings.llm)
    failures: list[str] = []

    # 1. Raw round trip. Isolates transport and credentials from the prompts.
    #    max_tokens covers reasoning as well as the answer on models that think
    #    by default, so keep it well above the length of the reply itself.
    print("\n[1/3] provider.complete()")
    try:
        reply = provider.complete(
            "Answer with a single word.", "Reply OK if you receive this.", max_tokens=256
        )
        print(f"      -> {reply.strip()!r}")
    except Exception as exc:  # noqa: BLE001 - report any provider failure verbatim
        print(f"      FAILED: {type(exc).__name__}: {exc}")
        print("\nThe provider is unreachable, so the stages below would only")
        print("exercise the fallback path. Fix this error first.")
        return 1

    service = RecommendationLLM(provider, settings.llm)

    # 2. Query understanding. The heuristic fallback cannot read Vietnamese or
    #    translate, so a rewritten `search` field proves the model ran.
    print("\n[2/3] understand_query()")
    parsed = service.understand_query(args.query)
    print(f"      input   : {args.query}")
    print(f"      search  : {parsed.get('search')}")
    print(f"      level   : {parsed.get('level')}")
    print(f"      category: {parsed.get('category')}")
    if parsed.get("search", "").strip() == args.query.strip():
        print("      NOTE: search is unchanged -> heuristic fallback, not the model")
        failures.append("understand_query")

    # 3. Explanation. The template always opens with this fixed phrase.
    print("\n[3/3] explain_recommendations()")
    explanation = service.explain_recommendations(args.query, COURSES)
    print(f"      {explanation[:300]}")
    if explanation.startswith("Based on your goal"):
        print("      NOTE: template fallback, not the model")
        failures.append("explain_recommendations")

    print("\n" + "=" * 62)
    if failures:
        print(f"FALLBACK USED BY: {', '.join(failures)}")
        print("The provider answered stage 1, so credentials work; the prompt")
        print("or the response parsing in src/llm/service.py is the suspect.")
        return 1
    print("OK - every stage answered by the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
