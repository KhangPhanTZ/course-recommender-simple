"""Typed loader for the YAML recommender config.

Kept tolerant: unknown keys are ignored and every non-essential field has a
sensible default, so older/newer config files keep working.
"""
from dataclasses import dataclass

import yaml


@dataclass
class Columns:
    id: str | None
    title: str
    description: str
    skills: str | None = None
    category: str | None = None
    level: str | None = None
    rating: str | None = None
    url: str | None = None


@dataclass
class Config:
    columns: Columns
    text_fields: list[str]
    min_characters: int = 30
    use_sbert: bool = True
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tfidf_max_features: int = 50000
    kmeans_k: int = 20
    random_state: int = 42
    top_k: int = 10
    overfetch: int = 50
    use_rerank: bool = False

    @property
    def backend(self) -> str:
        return "sbert" if self.use_sbert else "tfidf"


def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cols = raw.get("columns", {})
    columns = Columns(
        id=cols.get("id"),
        title=cols["title"],
        description=cols["description"],
        skills=cols.get("skills"),
        category=cols.get("category"),
        level=cols.get("level"),
        rating=cols.get("rating"),
        url=cols.get("url"),
    )

    return Config(
        columns=columns,
        text_fields=raw["text_fields"],
        min_characters=int(raw.get("min_characters", 30)),
        use_sbert=bool(raw.get("use_sbert", True)),
        sbert_model=raw.get("sbert_model", "sentence-transformers/all-MiniLM-L6-v2"),
        tfidf_max_features=int(raw.get("tfidf_max_features", 50000)),
        kmeans_k=int(raw.get("kmeans_k", 20)),
        random_state=int(raw.get("random_state", 42)),
        top_k=int(raw.get("top_k", 10)),
        overfetch=int(raw.get("overfetch", 50)),
        use_rerank=bool(raw.get("use_rerank", False)),
    )
