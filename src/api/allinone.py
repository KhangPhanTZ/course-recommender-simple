"""All-in-one ASGI app: serves the built web UI *and* the API from one origin.

Used by the self-contained demo image so a single service (one URL) hosts both
the SPA and the JSON API. The API is mounted under ``/api`` (matching the
frontend's default ``VITE_API_URL=/api``), and everything else falls back to the
SPA's ``index.html`` for client-side routing.

Run:  uvicorn src.api.allinone:app --host 0.0.0.0 --port 8000
Env:  STATIC_DIR  directory holding the built frontend (default: ``static``)
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .routes import router

logger = logging.getLogger("recsys.allinone")


def _resolve_static() -> str | None:
    """Find the built SPA directory, trying a few sensible locations."""
    candidates = [
        os.getenv("STATIC_DIR"),
        "static",
        "/app/static",
        os.path.join(os.path.dirname(__file__), "..", "..", "static"),
        os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"),
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "index.html")):
            return os.path.abspath(c)
    return None


def create_app() -> FastAPI:
    app = FastAPI(title="Course Recommender (all-in-one)", version=__version__)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    # API under /api so it never collides with SPA client routes (e.g. /map).
    app.include_router(router, prefix="/api")

    static = _resolve_static()
    if static:
        print(f"[allinone] serving SPA from {static}", flush=True)
        assets = os.path.join(static, "assets")
        if os.path.isdir(assets):
            app.mount("/assets", StaticFiles(directory=assets), name="assets")
        index = os.path.join(static, "index.html")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            # Serve a real static file if present (guarding path traversal),
            # else the SPA entrypoint for client-side routing.
            candidate = os.path.abspath(os.path.join(static, full_path))
            if full_path and candidate.startswith(static + os.sep) and os.path.isfile(candidate):
                return FileResponse(candidate)
            return FileResponse(index)
    else:
        print(
            f"[allinone] WARNING: no SPA build found (STATIC_DIR={os.getenv('STATIC_DIR')!r}); "
            "serving API only under /api",
            flush=True,
        )

    return app


app = create_app()
