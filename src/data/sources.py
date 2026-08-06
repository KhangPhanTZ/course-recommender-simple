"""Multi-platform course-dataset ingestion.

Different course catalogs (Coursera, Udemy, edX, …) ship wildly different column
layouts. This module normalizes each onto a single canonical schema so they can
be merged into one searchable catalog:

    id, title, provider, skills, category, level,
    rating, url, description, syllabus, source

Each supported dataset is described by a :class:`SourceSpec`: the columns that
*identify* it, and how to map those columns onto the canonical names. Detection
is by column signature (not filename), so a file is recognized however it's
named. Drop a supported CSV into ``data/`` and the pipeline auto-detects and
merges it — no config edits.

Supported out of the box (Kaggle):
    - coursera : ``course, partner, skills, certificatetype, level, rating``
    - udemy    : andrewmvd/udemy-courses  (``course_title, subject, level, url, …``)
    - edx      : imuhammad/edx-courses    (``title, institution, subject, Level,
                 course_description, course_syllabus, course_url, …``)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Canonical catalog columns, in order.
CANONICAL = [
    "id", "title", "provider", "skills", "category",
    "level", "rating", "url", "description", "syllabus", "source",
]

# Data columns a SourceSpec.mapping may populate (id/provider/source are set
# separately; provider from a constant or a column, source from the spec name).
_DATA_COLUMNS = ["title", "skills", "category", "level", "rating", "url", "description", "syllabus"]

# Free-text level labels -> canonical {beginner, intermediate, advanced}.
_LEVEL_ALIASES = {
    "beginner": "beginner",
    "beginner level": "beginner",
    "introductory": "beginner",
    "intro": "beginner",
    "basic": "beginner",
    "foundational": "beginner",
    "intermediate": "intermediate",
    "intermediate level": "intermediate",
    "advanced": "advanced",
    "advanced level": "advanced",
    "expert": "advanced",
    "expert level": "advanced",
}
# Labels that carry no usable difficulty signal -> left unset.
_LEVEL_NULLS = {"", "all", "all levels", "mixed", "not calibrated", "nan", "none"}


def normalize_level(value) -> str | None:
    """Map a free-text difficulty label onto a canonical level, or ``None``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    key = str(value).strip().lower()
    if key in _LEVEL_NULLS:
        return None
    return _LEVEL_ALIASES.get(key, key)


@dataclass(frozen=True)
class SourceSpec:
    """How to recognize and normalize one dataset onto the canonical schema."""

    name: str                            # canonical source id, e.g. "udemy"
    signature: tuple[str, ...]           # lowercased columns that identify the source
    mapping: dict[str, str] = field(default_factory=dict)  # canonical -> source column (lowercased)
    provider: str | None = None          # constant provider label, when the data has none
    provider_col: str | None = None      # else read provider from this column


# Detection order matters: earlier, more specific signatures win.
SPECS: list[SourceSpec] = [
    SourceSpec(
        name="edx",
        signature=("title", "institution", "course_description"),
        mapping={
            "title": "title",
            "category": "subject",
            "level": "level",
            "url": "course_url",
            "description": "course_description",
            "syllabus": "course_syllabus",
        },
        provider_col="institution",
    ),
    SourceSpec(
        name="udemy",
        signature=("course_title", "subject", "num_subscribers"),
        mapping={
            "title": "course_title",
            "category": "subject",
            "level": "level",
            "url": "url",
        },
        provider="Udemy",
    ),
    SourceSpec(
        name="coursera",
        signature=("course", "partner", "certificatetype"),
        mapping={
            "title": "course",
            "skills": "skills",
            "category": "certificatetype",
            "level": "level",
            "rating": "rating",
            "url": "url",
            # Coursera has no long description; skills stand in as the text field.
            "description": "skills",
        },
        provider_col="partner",
    ),
]


def _lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose column names are stripped and lower-cased."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def detect_spec(df: pd.DataFrame) -> SourceSpec | None:
    """Return the first :class:`SourceSpec` whose signature columns are all present."""
    cols = set(_lower_columns(df).columns)
    for spec in SPECS:
        if all(sig in cols for sig in spec.signature):
            return spec
    return None


def normalize_source(df: pd.DataFrame, spec: SourceSpec) -> pd.DataFrame:
    """Project a raw source frame onto the canonical schema (minus a global id)."""
    df = _lower_columns(df)
    out = pd.DataFrame(index=range(len(df)))

    for canon in _DATA_COLUMNS:
        src = spec.mapping.get(canon)
        out[canon] = df[src].to_numpy() if src and src in df.columns else pd.NA

    if spec.provider is not None:
        out["provider"] = spec.provider
    elif spec.provider_col and spec.provider_col in df.columns:
        out["provider"] = df[spec.provider_col].to_numpy()
    else:
        out["provider"] = pd.NA

    out["source"] = spec.name
    out["level"] = out["level"].map(normalize_level)

    # Title is the one required field.
    out["title"] = out["title"].astype("string").str.strip()
    out = out[out["title"].notna() & (out["title"] != "")]
    return out.reset_index(drop=True)


def discover(data_dir: str | Path) -> list[str]:
    """List candidate dataset files (``*.csv`` / ``*.parquet``) under a directory."""
    root = Path(data_dir)
    if not root.is_dir():
        return []
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in {".csv", ".parquet"})
    return [str(p) for p in files]


def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _dedup(catalog: pd.DataFrame) -> pd.DataFrame:
    """Drop exact (title, provider) repeats; keep genuine cross-platform variants."""
    key = pd.DataFrame({
        "t": catalog["title"].astype("string").str.strip().str.lower(),
        "p": catalog["provider"].astype("string").fillna("").str.strip().str.lower(),
    })
    return catalog.loc[~key.duplicated()].reset_index(drop=True)


def load_catalog(paths: str | list[str], *, strict: bool = False) -> pd.DataFrame:
    """Load and merge one or more course datasets into the canonical catalog.

    Unrecognized files are skipped (or raise when ``strict``). A fresh contiguous
    ``id`` is assigned after merge + dedup so it stays stable for the API.
    """
    if isinstance(paths, str):
        paths = [paths]

    frames: list[pd.DataFrame] = []
    for path in paths:
        raw = _read_any(path)
        spec = detect_spec(raw)
        if spec is None:
            if strict:
                raise ValueError(f"Unrecognized course dataset schema: {path}")
            continue
        frames.append(normalize_source(raw, spec))

    if not frames:
        raise ValueError(
            "No supported course datasets found. Expected a Coursera, Udemy, or "
            "edX CSV (see src/data/sources.py)."
        )

    catalog = pd.concat(frames, ignore_index=True)[CANONICAL[1:]]
    catalog = _dedup(catalog)
    catalog.insert(0, "id", range(len(catalog)))
    return catalog


def source_counts(catalog: pd.DataFrame) -> dict[str, int]:
    """Per-source row counts, for build metadata."""
    if "source" not in catalog.columns:
        return {}
    return {str(k): int(v) for k, v in catalog["source"].value_counts().items()}
