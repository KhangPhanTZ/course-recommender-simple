"""Fine-tune a Sentence-BERT bi-encoder on catalog query pairs.

Reads ``(query, positive)`` pairs (see :mod:`src.training.pairs`) and trains with
``MultipleNegativesRankingLoss`` — the standard objective for retrieval when you
have positives but no hand-labelled negatives (other positives in the batch act
as negatives). The result is a drop-in replacement for the base SBERT model:
point ``sbert_model`` in ``config/config.yaml`` at the output directory, rebuild
artifacts, and the serving stack uses it unchanged.

``sentence-transformers`` / ``torch`` are imported lazily, so importing this
module stays cheap; training itself is meant to run on a GPU box or Colab (see
``notebooks/finetune_colab.ipynb``). Install the training deps with
``pip install -r requirements-train.txt``.
"""
from __future__ import annotations

import argparse

from .pairs import read_jsonl


def train(
    pairs_path: str,
    *,
    base_model: str = "BAAI/bge-small-en-v1.5",
    out_dir: str = "artifacts/finetuned-encoder",
    epochs: int = 1,
    batch_size: int = 32,
    warmup_ratio: float = 0.1,
) -> str:
    """Fine-tune ``base_model`` on the pairs and save to ``out_dir``.

    Returns the output directory. Raises ``ImportError`` with install guidance if
    the training stack is missing, and ``ValueError`` if there are no usable pairs.
    """
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except Exception as e:  # pragma: no cover - heavy optional deps
        raise ImportError(
            "Training needs sentence-transformers + torch. "
            "Install them with `pip install -r requirements-train.txt` "
            "(a GPU, e.g. Google Colab, is strongly recommended)."
        ) from e

    pairs = read_jsonl(pairs_path)
    examples = [
        InputExample(texts=[p["query"], p["positive"]])
        for p in pairs
        if p.get("query") and p.get("positive")
    ]
    if not examples:
        raise ValueError(f"No usable (query, positive) pairs in {pairs_path!r}.")

    model = SentenceTransformer(base_model)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = int(len(loader) * epochs * warmup_ratio)

    print(
        f"Fine-tuning {base_model} on {len(examples)} pairs "
        f"({epochs} epoch(s), batch {batch_size}, warmup {warmup})…"
    )
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup,
        show_progress_bar=True,
    )
    model.save(out_dir)
    print(f"Saved fine-tuned encoder to {out_dir!r}")
    print(
        "Next: set `sbert_model: " + out_dir + "` in config/config.yaml, "
        "rebuild (`python -m src.pipeline --mode build --data data/`), "
        "then compare metrics with `python scripts/run_eval.py`."
    )
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the retrieval bi-encoder.")
    ap.add_argument("--pairs", default="data/finetune/pairs.jsonl", help="JSONL from generate_pairs.py")
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5", help="Base model id or path")
    ap.add_argument("--out", default="artifacts/finetuned-encoder", help="Output directory")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    train(args.pairs, base_model=args.base, out_dir=args.out, epochs=args.epochs, batch_size=args.batch)


if __name__ == "__main__":
    main()
