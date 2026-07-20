"""Artifact storage abstraction.

Defines a minimal interface used by the pipeline and the serving layer to
persist and load model artifacts (vectors, indexes, dataframes) without
caring whether they live on the local filesystem or in the cloud (S3).

This is the seam that lets the exact same training/serving code run on a
laptop and on AWS ECS/Fargate — only the ``ARTIFACT_STORE`` env var changes.
"""
from __future__ import annotations

import abc
from typing import BinaryIO


class ArtifactStore(abc.ABC):
    """Read/write binary artifacts identified by a relative key.

    Keys look like ``"faiss.index"`` or ``"courses.parquet"``. Implementations
    map that key onto a concrete location (a directory, an S3 prefix, ...).
    """

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if an artifact with ``key`` is present."""

    @abc.abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """Return the full contents of ``key`` as bytes."""

    @abc.abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None:
        """Persist ``data`` under ``key``, overwriting any existing value."""

    @abc.abstractmethod
    def open_read(self, key: str) -> BinaryIO:
        """Return a binary file-like object positioned at the start of ``key``.

        Callers are responsible for closing it. Used for large artifacts
        (numpy/faiss) that libraries prefer to stream from a file handle.
        """

    @abc.abstractmethod
    def local_path(self, key: str) -> str:
        """Return a path on the local disk that contains ``key``.

        For cloud backends this downloads to a cache dir on first use. This
        exists because some libraries (faiss, numpy.load with mmap) need a
        real filesystem path rather than a byte stream.
        """

    def download_to(self, key: str, dest_path: str) -> str:
        """Copy ``key`` to ``dest_path`` on the local filesystem."""
        import os

        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(self.read_bytes(key))
        return dest_path
