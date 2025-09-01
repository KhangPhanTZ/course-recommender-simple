import re
import pandas as pd
from typing import List

PUNCT_RE = re.compile(r"[^a-zA-Z0-9\s\-\+\#]")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_for_model(s: str) -> str:
    s = s.lower()
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_text_row(row: pd.Series, fields: List[str]) -> str:
    parts = []
    for f in fields:
        val = row.get(f, "")
        parts.append(clean_text(str(val)))
    joined = " | ".join([p for p in parts if p])
    return normalize_for_model(joined)
