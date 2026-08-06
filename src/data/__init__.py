"""Dataset ingestion: normalize heterogeneous course catalogs into one schema."""
from .sources import (
    CANONICAL,
    SPECS,
    SourceSpec,
    detect_spec,
    discover,
    load_catalog,
    normalize_level,
    normalize_source,
    source_counts,
)

__all__ = [
    "CANONICAL",
    "SPECS",
    "SourceSpec",
    "detect_spec",
    "discover",
    "load_catalog",
    "normalize_level",
    "normalize_source",
    "source_counts",
]
