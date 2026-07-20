import pytest

from src.storage import LocalArtifactStore, get_artifact_store


def test_local_store_roundtrip(tmp_path):
    store = LocalArtifactStore(base_dir=str(tmp_path))
    assert store.exists("x.bin") is False
    store.write_bytes("x.bin", b"hello")
    assert store.exists("x.bin") is True
    assert store.read_bytes("x.bin") == b"hello"
    assert store.local_path("x.bin").endswith("x.bin")


def test_local_store_nested_keys(tmp_path):
    store = LocalArtifactStore(base_dir=str(tmp_path))
    store.write_bytes("sub/dir/y.bin", b"1234")
    assert store.read_bytes("sub/dir/y.bin") == b"1234"


def test_get_artifact_store_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACT_STORE", "local")
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    store = get_artifact_store()
    assert isinstance(store, LocalArtifactStore)


def test_s3_requires_bucket(monkeypatch):
    monkeypatch.setenv("ARTIFACT_STORE", "s3")
    monkeypatch.delenv("ARTIFACT_S3_BUCKET", raising=False)
    with pytest.raises(ValueError):
        get_artifact_store()


def test_unknown_store(monkeypatch):
    monkeypatch.setenv("ARTIFACT_STORE", "gcs")
    with pytest.raises(ValueError):
        get_artifact_store()
