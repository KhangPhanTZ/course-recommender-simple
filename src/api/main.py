"""FastAPI application factory for the course recommender service.

Run locally::

    uvicorn src.api.main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive OpenAPI docs.
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from .routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("recsys.api")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Course Recommender API",
        description=(
            "Content-based course recommendations with a two-stage retriever "
            "(ANN + cross-encoder) and a GenAI/RAG layer for query understanding "
            "and explanations. Deployable to AWS ECS/Fargate."
        ),
        version=__version__,
    )

    # CORS: open by default for the demo; lock down via env in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.1f}"
        logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms)
        return response

    app.include_router(router)

    @app.get("/", tags=["ops"])
    def root():
        return {"service": "course-recommender", "version": __version__, "docs": "/docs"}

    return app


app = create_app()
