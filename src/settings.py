"""Application-wide runtime settings, driven by environment variables.

Kept intentionally dependency-light (stdlib only) so every module — including
the lightweight CI import checks — can import it without pulling in pydantic,
boto3 or the ML stack.

Load order: process environment wins; a local ``.env`` file (if present) is
used as a fallback for developer convenience.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _load_dotenv(path: str = ".env") -> None:
    """Populate ``os.environ`` from a ``.env`` file without overriding real env.

    Minimal parser (no external dependency): ``KEY=VALUE`` lines, ``#`` comments.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
    except OSError:
        pass


def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class LLMSettings:
    """Configuration for the GenAI / RAG layer."""

    #: "anthropic" (Claude API), "bedrock" (AWS Bedrock), or "disabled".
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    #: Model id. For Anthropic e.g. "claude-sonnet-5"; for Bedrock the ARN/id.
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-5"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "512")))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2")))
    #: AWS region for Bedrock.
    aws_region: str | None = field(default_factory=lambda: os.getenv("AWS_REGION"))
    #: Turn the whole layer off (falls back to deterministic templates).
    enabled: bool = field(default_factory=lambda: _get_bool("LLM_ENABLED", True))

    @property
    def is_active(self) -> bool:
        return self.enabled and self.provider != "disabled"


@dataclass
class AppSettings:
    """Top-level settings object shared across API and pipeline."""

    config_path: str = field(default_factory=lambda: os.getenv("RECSYS_CONFIG", "config/config.yaml"))
    artifact_store: str = field(default_factory=lambda: os.getenv("ARTIFACT_STORE", "local"))
    #: Top-N returned by default.
    default_top_k: int = field(default_factory=lambda: int(os.getenv("DEFAULT_TOP_K", "10")))
    #: Enable cross-encoder reranking in the API (can be heavy).
    enable_rerank: bool = field(default_factory=lambda: _get_bool("ENABLE_RERANK", True))
    llm: LLMSettings = field(default_factory=LLMSettings)


def get_settings() -> AppSettings:
    """Return a freshly-resolved :class:`AppSettings` (reads .env once)."""
    _load_dotenv()
    return AppSettings()
