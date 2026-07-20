"""GenAI / RAG layer: provider abstraction + query-understanding & explanations."""
from .provider import LLMProvider, get_provider
from .service import RecommendationLLM

__all__ = ["LLMProvider", "get_provider", "RecommendationLLM"]
