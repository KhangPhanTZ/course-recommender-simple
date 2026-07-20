"""Pluggable artifact storage (local filesystem or AWS S3).

Usage::

    from src.storage import get_artifact_store
    store = get_artifact_store()          # from env
    store.write_bytes("courses.parquet", data)

Selection is driven by environment variables so the same code runs locally
and on AWS:

- ``ARTIFACT_STORE``: ``"local"`` (default) or ``"s3"``.
- ``ARTIFACT_DIR``: base dir for the local store (default ``artifacts``).
- ``ARTIFACT_S3_BUCKET`` / ``ARTIFACT_S3_PREFIX``: S3 target.
- ``AWS_REGION``: region for the S3 client.
"""
from __future__ import annotations

import os

from .base import ArtifactStore
from .local import LocalArtifactStore
from .s3 import S3ArtifactStore

__all__ = [
    "ArtifactStore",
    "LocalArtifactStore",
    "S3ArtifactStore",
    "get_artifact_store",
]


def get_artifact_store(kind: str | None = None) -> ArtifactStore:
    """Build an :class:`ArtifactStore` from the environment (or ``kind``)."""
    kind = (kind or os.getenv("ARTIFACT_STORE", "local")).lower()

    if kind == "local":
        return LocalArtifactStore(base_dir=os.getenv("ARTIFACT_DIR", "artifacts"))

    if kind == "s3":
        bucket = os.getenv("ARTIFACT_S3_BUCKET")
        if not bucket:
            raise ValueError(
                "ARTIFACT_STORE=s3 requires ARTIFACT_S3_BUCKET to be set."
            )
        return S3ArtifactStore(
            bucket=bucket,
            prefix=os.getenv("ARTIFACT_S3_PREFIX", "artifacts"),
            region=os.getenv("AWS_REGION"),
        )

    raise ValueError(f"Unknown ARTIFACT_STORE kind: {kind!r} (use 'local' or 's3').")
