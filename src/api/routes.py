"""API routes for the course recommender service."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from .. import __version__
from ..llm.tracks import assemble_roadmap, list_tracks
from . import deps
from .schemas import (
    ChatRequest,
    ChatResponse,
    CourseHit,
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    Roadmap,
    RoadmapRequest,
    SimilarRequest,
    TrackInfo,
)

logger = logging.getLogger("recsys.api")
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness/readiness probe. Reports whether artifacts and LLM are ready."""
    s = deps.settings()
    backend = n = None
    try:
        rec = deps.get_recommender()
        backend, n = rec.backend, rec.size
        status = "ok"
    except Exception as exc:  # artifacts not built yet -> degraded, not dead
        logger.warning("Recommender not ready: %s", exc)
        status = "degraded"
    return HealthResponse(
        status=status,
        backend=backend,
        n_courses=n,
        llm_provider=s.llm.provider if s.llm.is_active else None,
        llm_enabled=s.llm.is_active,
        version=__version__,
    )


@router.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(req: RecommendRequest) -> RecommendResponse:
    """Recommend courses for a free-text goal, optionally with RAG explanation."""
    rec = deps.get_recommender()
    llm = deps.get_llm()
    s = deps.settings()

    resolved = req.query
    filters = {}
    if req.level:
        filters["level"] = req.level
    if req.category:
        filters["category"] = req.category

    # LLM/heuristic query understanding fills in filters not set explicitly.
    if req.understand:
        intent = llm.understand_query(req.query)
        resolved = intent.get("search") or req.query
        for key in ("level", "category"):
            if intent.get(key) and not filters.get(key):
                filters[key] = intent[key]
    filters = {k: v for k, v in filters.items() if v}

    use_rerank = s.enable_rerank if req.rerank is None else req.rerank
    hits = rec.recommend(
        resolved, top_k=req.top_k, filters=filters, use_rerank=use_rerank
    )
    results = [CourseHit(**_hit(h)) for h in hits]

    explanation = None
    if req.explain:
        explanation = llm.explain_recommendations(req.query, [h.to_dict() for h in hits])

    return RecommendResponse(
        query=req.query,
        resolved_query=resolved,
        filters=filters,
        results=results,
        explanation=explanation,
        llm_enabled=llm.enabled,
    )


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest) -> ChatResponse:
    """Conversational advisor: grounds a reply in courses retrieved for the last message."""
    rec = deps.get_recommender()
    llm = deps.get_llm()

    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    hits = rec.recommend(last_user, top_k=req.top_k) if last_user.strip() else []
    history = [{"role": m.role, "content": m.content} for m in req.messages]
    reply = llm.advise_chat(history, [h.to_dict() for h in hits])

    return ChatResponse(
        reply=reply,
        courses=[CourseHit(**_hit(h)) for h in hits],
        llm_enabled=llm.enabled,
    )


@router.get("/roadmap/tracks", response_model=list[TrackInfo], tags=["roadmap"])
def roadmap_tracks() -> list[TrackInfo]:
    """List the curated career tracks available for a roadmap."""
    return [TrackInfo(**t) for t in list_tracks()]


@router.post("/roadmap", response_model=Roadmap, tags=["roadmap"])
def roadmap(req: RoadmapRequest) -> Roadmap:
    """Build a grounded, tiered learning roadmap for a career track."""
    rec = deps.get_recommender()
    llm = deps.get_llm()

    def retrieve(query: str, k: int) -> list[dict]:
        return [_hit(h) for h in rec.recommend(query, top_k=k)]

    data = assemble_roadmap(req.track, retrieve, per_tier=req.per_tier)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Unknown track: {req.track}")

    data["intro"] = llm.advise_roadmap(data)
    data["llm_enabled"] = llm.enabled
    return Roadmap(**data)


@router.post("/similar", response_model=list[CourseHit], tags=["recommend"])
def similar(req: SimilarRequest) -> list[CourseHit]:
    """Find courses similar to an existing course id."""
    rec = deps.get_recommender()
    try:
        hits = rec.similar_to(req.course_id, top_k=req.top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [CourseHit(**_hit(h)) for h in hits]


@router.get("/courses/{course_id}", tags=["catalog"])
def get_course(course_id: str):
    """Fetch a single course by id (tries int then string)."""
    rec = deps.get_recommender()
    for cid in _id_candidates(course_id):
        course = rec.get_course(cid)
        if course is not None:
            return course
    raise HTTPException(status_code=404, detail=f"Course not found: {course_id}")


def _hit(reco) -> dict:
    d = reco.to_dict()
    d.pop("explanation", None)
    return d


def _id_candidates(raw: str):
    try:
        yield int(raw)
    except ValueError:
        pass
    yield raw
