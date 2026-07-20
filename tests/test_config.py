import yaml

from src.utils.config import load_config


def test_load_config_defaults(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "columns": {"title": "course", "description": "partner"},
        "text_fields": ["title"],
    }))
    cfg = load_config(str(cfg_path))
    assert cfg.columns.title == "course"
    assert cfg.use_sbert is True            # default
    assert cfg.top_k == 10                  # default
    assert cfg.backend == "sbert"


def test_repo_config_is_single_backend():
    """Regression: config.yaml must not define use_sbert twice (last-wins bug)."""
    with open("config/config.yaml") as f:
        lines = f.read().splitlines()
    n = sum(1 for ln in lines if ln.strip().startswith("use_sbert:"))
    assert n == 1, f"expected exactly one 'use_sbert:' key, found {n}"
    cfg = load_config("config/config.yaml")
    assert cfg.backend in ("sbert", "tfidf")
