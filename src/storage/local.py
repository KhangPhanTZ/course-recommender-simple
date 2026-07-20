"""Local filesystem implementation of :class:`ArtifactStore`."""
from __future__ import annotations

import os
import shutil
from typing import BinaryIO

from .base import ArtifactStore


class LocalArtifactStore(ArtifactStore):
    """Store artifacts under a base directory on the local disk.

    Parameters
    ----------
    base_dir:
        Root directory. Keys are resolved relative to it, e.g. key
        ``"faiss.index"`` -> ``<base_dir>/faiss.index``.
    """

    def __init__(self, base_dir: str = "artifacts") -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.base_dir, key)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def read_bytes(self, key: str) -> bytes:
        with open(self._path(key), "rb") as f:
            return f.read()

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def open_read(self, key: str) -> BinaryIO:
        return open(self._path(key), "rb")

    def local_path(self, key: str) -> str:
        path = self._path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact not found: {key} ({path})")
        return path

    def download_to(self, key: str, dest_path: str) -> str:
        src = self.local_path(key)
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        if os.path.abspath(src) != os.path.abspath(dest_path):
            shutil.copyfile(src, dest_path)
        return dest_path
