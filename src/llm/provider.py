"""LLM provider abstraction for the GenAI / RAG layer.

Two interchangeable backends behind one interface so the same application
code runs against a hosted API or fully inside AWS:

- :class:`AnthropicProvider` — the Claude API (``anthropic`` SDK).
- :class:`BedrockProvider`   — Claude on **AWS Bedrock** (``boto3``), the
  cloud-native path for the AWS deployment.

SDKs are imported lazily so importing this module never requires them. A
:class:`NullProvider` keeps the system fully functional (deterministic
template output) when no LLM is configured or a call fails.
"""
from __future__ import annotations

import abc
import json

from ..settings import LLMSettings


class LLMProvider(abc.ABC):
    """Minimal chat-completion interface used by the recommender's GenAI layer."""

    @abc.abstractmethod
    def complete(self, system: str, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Return the model's text completion for ``prompt`` under ``system``."""

    @property
    def available(self) -> bool:  # pragma: no cover - trivial
        return True


class NullProvider(LLMProvider):
    """No-op provider: signals unavailability so callers use fallbacks."""

    def complete(self, system: str, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        raise RuntimeError("LLM is disabled (NullProvider).")

    @property
    def available(self) -> bool:
        return False


class AnthropicProvider(LLMProvider):
    """Claude via the Anthropic API. Requires ``ANTHROPIC_API_KEY`` in env."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic  # lazy

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def complete(self, system: str, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        # ``temperature`` is accepted for interface compatibility but not sent:
        # current Claude models reject sampling parameters with a 400.
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")


class BedrockProvider(LLMProvider):
    """Claude via AWS Bedrock runtime (Messages API). Uses IAM credentials."""

    def __init__(self, model: str, region: str | None = None) -> None:
        self.model = model
        self.region = region
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3  # lazy

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def complete(self, system: str, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        # ``temperature`` omitted for the same reason as AnthropicProvider.complete.
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        resp = self.client.invoke_model(modelId=self.model, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        return "".join(
            block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
        )


def get_provider(settings: LLMSettings | None = None) -> LLMProvider:
    """Build an :class:`LLMProvider` from :class:`LLMSettings` (or the env)."""
    settings = settings or LLMSettings()
    if not settings.is_active:
        return NullProvider()

    provider = settings.provider.lower()
    if provider == "anthropic":
        return AnthropicProvider(model=settings.model)
    if provider == "bedrock":
        return BedrockProvider(model=settings.model, region=settings.aws_region)
    if provider == "disabled":
        return NullProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.provider!r}")
