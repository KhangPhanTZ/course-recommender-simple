"""Generate fine-tuning pairs from the catalog and write them as JSON Lines.

Loads courses from the built artifacts (``courses.parquet`` via the same
ArtifactStore the API uses), then emits ``(query, positive)`` pairs for
fine-tuning the retrieval encoder (see :mod:`src.training.pairs`).

Examples::

    # offline, deterministic — no API key needed (good default / CI)
    python scripts/generate_pairs.py --mode proxy

    # synthetic queries via the configured LLM (needs ANTHROPIC_API_KEY)
    python scripts/generate_pairs.py --mode llm --per-course 3 --limit 500

The catalog comes from artifacts, so run a build first
(``python -m src.pipeline --mode build --data data/``).
"""
from __future__ import annotations

import argparse
import io
import sys

sys.path.insert(0, ".")

import pandas as pd  # noqa: E402

from src.storage import get_artifact_store  # noqa: E402
from src.training.pairs import build_llm_pairs, build_proxy_pairs, write_jsonl  # noqa: E402


def _load_courses() -> pd.DataFrame:
    store = get_artifact_store()
    if not store.exists("courses.parquet"):
        raise SystemExit(
            "courses.parquet not found. Build artifacts first:\n"
            "  python -m src.pipeline --mode build --data data/"
        )
    return pd.read_parquet(io.BytesIO(store.read_bytes("courses.parquet")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate fine-tuning pairs from the catalog.")
    ap.add_argument("--mode", choices=["proxy", "llm"], default="proxy")
    ap.add_argument("--out", default="data/finetune/pairs.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="Max courses to use (0 = all)")
    ap.add_argument("--per-course", type=int, default=3, help="Queries per course (llm mode)")
    args = ap.parse_args()

    df = _load_courses()
    if args.limit > 0:
        df = df.head(args.limit)
    records = df.to_dict("records")
    print(f"Loaded {len(records)} courses from artifacts.")

    if args.mode == "proxy":
        pairs = build_proxy_pairs(records)
    else:
        from src.llm.provider import get_provider

        provider = get_provider()
        if not provider.available:
            raise SystemExit(
                "LLM mode needs an available provider (set ANTHROPIC_API_KEY, "
                "LLM_PROVIDER=anthropic). Use --mode proxy for the offline path."
            )
        pairs = build_llm_pairs(records, provider.complete, per_course=args.per_course)

    n = write_jsonl(pairs, args.out)
    print(f"Wrote {n} '{args.mode}' pairs to {args.out}")


if __name__ == "__main__":
    main()
