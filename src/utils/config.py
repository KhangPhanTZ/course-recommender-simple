import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Columns:
    id: Optional[str]
    title: str
    description: str
    skills: Optional[str]
    category: Optional[str]
    level: Optional[str]
    rating: Optional[str]
    url: Optional[str]

@dataclass
class Config:
    columns: Columns
    text_fields: List[str]
    min_characters: int
    use_sbert: bool
    sbert_model: str
    kmeans_k: int
    random_state: int
    top_k: int

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
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
        min_characters=int(raw["min_characters"]),
        use_sbert=bool(raw["use_sbert"]),
        sbert_model=raw["sbert_model"],
        kmeans_k=int(raw["kmeans_k"]),
        random_state=int(raw["random_state"]),
        top_k=int(raw["top_k"]),
    )
