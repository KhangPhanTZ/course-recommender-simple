"""Text vectorization: corpus construction, TF-IDF, and Sentence-BERT.

The serving-time similarity search lives in :mod:`src.recsys.recommender` and
:mod:`src.recsys.index`; this module only produces the vectors the pipeline
persists as artifacts.
"""
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from ..fe_text import build_text_row


def build_corpus(df: pd.DataFrame, text_fields: list[str], min_chars: int) -> tuple[pd.DataFrame, list[str]]:
    """Join the configured text fields per row and drop too-short documents."""
    df = df.copy()
    df["__text__"] = df.apply(lambda r: build_text_row(r, text_fields), axis=1)
    df = df[df["__text__"].str.len() >= min_chars].reset_index(drop=True)
    corpus = df["__text__"].tolist()
    return df, corpus


def fit_tfidf(corpus, max_features: int = 50000):
    """Fit a TF-IDF vectorizer with domain stopwords and 1-2 grams."""
    custom = {
        "course", "courses", "introduction", "intro", "learn", "learning",
        "beginner", "beginners", "for", "with", "using", "and", "the",
    }
    stop = list(ENGLISH_STOP_WORDS.union(custom))  # sklearn wants a list, not a frozenset

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words=stop,
        max_df=0.85,
        min_df=2,
        token_pattern=r"(?u)\b[^\d\W][\w\-]+\b",
    )
    X = vec.fit_transform(corpus)
    return vec, X


def embed_sbert(corpus: list[str], model_name: str):
    """Encode the corpus into normalized dense Sentence-BERT vectors."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover - optional heavy dep
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run `pip install sentence-transformers` to use the SBERT backend."
        ) from e
    model = SentenceTransformer(model_name)
    X = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return model, X
