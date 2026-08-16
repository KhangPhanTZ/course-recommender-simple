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
    url_direct: bool = False
    skills: str | None = None
    provider: str | None = None
    source: str | None = None


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


class ChatMessage(BaseModel):
    role: str = Field(..., description='"user" or "assistant".')
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    top_k: int = Field(6, ge=1, le=20)

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "user", "content": "I want to move into data engineering — where do I start?"}
                ]
            }
        }
    }


class ChatResponse(BaseModel):
    reply: str
    courses: list[CourseHit]
    llm_enabled: bool = False


class CatalogCourse(BaseModel):
    id: Any
    title: str
    provider: str | None = None
    source: str | None = None
    category: str | None = None
    level: str | None = None
    rating: Any | None = None
    url: str | None = None
    url_direct: bool = False
    skills: str | None = None
    description: str | None = None


class CatalogResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[CatalogCourse]


class TrackInfo(BaseModel):
    id: str
    label: str
    summary: str
    group: str = ""


class RoadmapNode(BaseModel):
    id: str
    tier: str
    skills: list[str] = Field(default_factory=list)
    courses: list[CourseHit] = Field(default_factory=list)


class RoadmapEdge(BaseModel):
    source: str
    target: str
    kind: str = "progress"


class RoadmapBridge(BaseModel):
    track: str
    label: str
    note: str


class Roadmap(BaseModel):
    track: str
    label: str
    summary: str
    intro: str | None = None
    nodes: list[RoadmapNode] = Field(default_factory=list)
    edges: list[RoadmapEdge] = Field(default_factory=list)
    bridges: list[RoadmapBridge] = Field(default_factory=list)
    llm_enabled: bool = False


class RoadmapRequest(BaseModel):
    track: str = Field(..., description="Career track id, e.g. 'ml-engineer'.")
    per_tier: int = Field(2, ge=1, le=5, description="Courses to surface per tier.")


class HealthResponse(BaseModel):
    status: str
    backend: str | None = None
    n_courses: int | None = None
    llm_provider: str | None = None
    llm_enabled: bool = False
    version: str
