"""Lazy, cached singletons for the API (recommender + LLM).

Artifacts and models are loaded once on first use and reused across requests.
Loading is deferred (not at import time) so the process starts fast and the
container can pass health checks while a large model warms up.
"""
from __future__ import annotations

import threading

from ..llm import RecommendationLLM
from ..recsys.recommender import Recommender
from ..settings import AppSettings, get_settings
from ..storage import get_artifact_store

_lock = threading.Lock()
_recommender: Recommender | None = None
_llm: RecommendationLLM | None = None
_settings: AppSettings | None = None


def settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def get_recommender() -> Recommender:
    """Return the process-wide :class:`Recommender`, loading it on first call."""
    global _recommender
    if _recommender is None:
        with _lock:
            if _recommender is None:
                _recommender = Recommender.load(get_artifact_store())
    return _recommender


def get_llm() -> RecommendationLLM:
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                _llm = RecommendationLLM(settings=settings().llm)
    return _llm


def reset_state() -> None:
    """Clear cached singletons (used in tests)."""
    global _recommender, _llm, _settings
    with _lock:
        _recommender = None
        _llm = None
        _settings = None
