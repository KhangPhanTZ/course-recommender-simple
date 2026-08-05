"""All-in-one ASGI app: serves the built web UI *and* the API from one origin.

Used by the self-contained demo image so a single service (one URL) hosts both
the SPA and the JSON API. The API is mounted under ``/api`` (matching the
frontend's default ``VITE_API_URL=/api``), and everything else falls back to the
SPA's ``index.html`` for client-side routing.

Run:  uvicorn src.api.allinone:app --host 0.0.0.0 --port 8000
Env:  STATIC_DIR  directory holding the built frontend (default: ``static``)
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .routes import router

STATIC_DIR = os.getenv("STATIC_DIR", "static")


def create_app() -> FastAPI:
    app = FastAPI(title="Course Recommender (all-in-one)", version=__version__)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    # API under /api so it never collides with SPA client routes (e.g. /map).
    app.include_router(router, prefix="/api")

    if os.path.isdir(STATIC_DIR):
        assets = os.path.join(STATIC_DIR, "assets")
        if os.path.isdir(assets):
            app.mount("/assets", StaticFiles(directory=assets), name="assets")
        index = os.path.join(STATIC_DIR, "index.html")

        base = os.path.abspath(STATIC_DIR)

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            # Serve a real static file if present (guarding against path
            # traversal), else the SPA entrypoint for client-side routing.
            candidate = os.path.abspath(os.path.join(STATIC_DIR, full_path))
            if full_path and candidate.startswith(base + os.sep) and os.path.isfile(candidate):
                return FileResponse(candidate)
            return FileResponse(index)

    return app


app = create_app()
