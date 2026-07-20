"""AWS S3 implementation of :class:`ArtifactStore`.

``boto3`` is imported lazily so that importing this module (and running the
lightweight CI import checks) does not require the AWS SDK to be installed.
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import BinaryIO

from .base import ArtifactStore


class S3ArtifactStore(ArtifactStore):
    """Store artifacts in an S3 bucket under an optional key prefix.

    Parameters
    ----------
    bucket:
        Target S3 bucket name.
    prefix:
        Optional key prefix, e.g. ``"artifacts/v1"``. The full object key is
        ``<prefix>/<key>``.
    cache_dir:
        Local directory used by :meth:`local_path` to cache downloads so that
        repeated reads (and libraries that need a real path) are cheap.
    region:
        AWS region. Falls back to the SDK's default resolution chain.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        cache_dir: str | None = None,
        region: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.cache_dir = cache_dir or os.path.join(
            tempfile.gettempdir(), "recsys-artifacts"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self._region = region
        self._client = None  # lazily created

    @property
    def client(self):
        if self._client is None:
            import boto3  # lazy import: keeps AWS SDK optional

            self._client = boto3.client("s3", region_name=self._region)
        return self._client

    def _object_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._object_key(key))
            return True
        except ClientError as exc:  # pragma: no cover - network path
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def read_bytes(self, key: str) -> bytes:
        buf = io.BytesIO()
        self.client.download_fileobj(self.bucket, self._object_key(key), buf)
        return buf.getvalue()

    def write_bytes(self, key: str, data: bytes) -> None:
        self.client.upload_fileobj(
            io.BytesIO(data), self.bucket, self._object_key(key)
        )

    def open_read(self, key: str) -> BinaryIO:
        return open(self.local_path(key), "rb")

    def local_path(self, key: str) -> str:
        dest = os.path.join(self.cache_dir, key)
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
            self.client.download_file(self.bucket, self._object_key(key), dest)
        return dest
