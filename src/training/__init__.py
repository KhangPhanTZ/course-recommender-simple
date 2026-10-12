"""Fine-tuning the retrieval bi-encoder on the course catalog.

The serving stack uses a *pre-trained* Sentence-BERT model out of the box. This
package adds an optional step that adapts (fine-tunes) that model to the catalog
domain, so retrieval ranks course text more faithfully for learner queries.

Because there is no real click data, training pairs are built from the catalog
itself — either synthetically via an LLM (``build_llm_pairs``, InPars-style) or
deterministically from the course fields (``build_proxy_pairs``). Both produce
``(query, positive_text)`` pairs suitable for ``MultipleNegativesRankingLoss``.

The pair-building logic lives in :mod:`src.training.pairs` (pure, testable, no
heavy deps); the actual training lives in :mod:`src.training.finetune` (imports
``sentence-transformers``/``torch`` lazily, meant to run on a GPU box / Colab).
"""
