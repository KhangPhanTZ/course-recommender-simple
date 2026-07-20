"""Pydantic request/response models for the recommender API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Free-text learning goal.")
    top_k: int = Field(10, ge=1, le=100)
    level: str | None = Field(None, description="Filter: beginner/intermediate/advanced.")
    category: str | None = Field(None, description="Filter: course category.")
    rerank: bool | None = Field(None, description="Override cross-encoder reranking.")
    explain: bool = Field(False, description="Add an LLM-generated rationale (RAG).")
    understand: bool = Field(True, description="Use LLM query understanding to extract filters.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "I want to learn deep learning with pytorch as a beginner",
                "top_k": 5,
                "explain": True,
            }
        }
    }


class CourseHit(BaseModel):
    id: Any
    title: str
    score: float
    category: str | None = None
    level: str | None = None
    rating: Any | None = None
    url: str | None = None
    skills: str | None = None


class RecommendResponse(BaseModel):
    query: str
    resolved_query: str = Field(..., description="Query after LLM/heuristic understanding.")
    filters: dict = Field(default_factory=dict)
    results: list[CourseHit]
    explanation: str | None = None
    llm_enabled: bool = False


class SimilarRequest(BaseModel):
    course_id: Any = Field(..., description="Existing course id to find neighbours for.")
    top_k: int = Field(10, ge=1, le=100)


class HealthResponse(BaseModel):
    status: str
    backend: str | None = None
    n_courses: int | None = None
    llm_provider: str | None = None
    llm_enabled: bool = False
    version: str
