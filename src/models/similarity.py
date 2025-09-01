import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy import sparse
from . import vector_backend
from ..utils.io import save_pickle, save_sparse_matrix, save_parquet
from ..fe_text import build_text_row

def build_corpus(df: pd.DataFrame, text_fields: List[str], min_chars: int) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df["__text__"] = df.apply(lambda r: build_text_row(r, text_fields), axis=1)
    df = df[df["__text__"].str.len() >= min_chars].reset_index(drop=True)
    corpus = df["__text__"].tolist()
    return df, corpus

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

def fit_tfidf(corpus, max_features: int = 50000):
    custom = {
        "course","courses","introduction","intro","learn","learning",
        "beginner","beginners","for","with","using","and","the"
    }
    stop = list(ENGLISH_STOP_WORDS.union(custom))   # <<< ép về list

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words=stop,          # giờ là list
        max_df=0.85,
        min_df=2,
        token_pattern=r"(?u)\b[^\d\W][\w\-]+\b"
    )
    X = vec.fit_transform(corpus)
    return vec, X


def embed_sbert(corpus: List[str], model_name: str):
    model = SentenceTransformer(model_name)
    X = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return model, X

def build_similarity_artifacts(df: pd.DataFrame, corpus: List[str], use_sbert: bool, model_name: str):
    if use_sbert:
        model, X = embed_sbert(corpus, model_name)
        save_pickle(model_name, "artifacts/sbert_model_name.pkl")  # store name only
        np.save("artifacts/X_sbert.npy", X)
    else:
        vec, X = fit_tfidf(corpus)
        save_pickle(vec, "artifacts/tfidf_vectorizer.pkl")
        save_sparse_matrix(X, "artifacts/X_tfidf.npz")
    save_parquet(df.drop(columns=["__text__"]), "artifacts/courses.parquet")

def topk_similar_by_index(idx: int, X, top_k: int = 10, exclude_self: bool = True) -> List[Tuple[int, float]]:
    if sparse.issparse(X):
        sims = cosine_similarity(X[idx], X).ravel()
    else:
        sims = np.dot(X[idx], X.T)  # assuming normalized if SBERT
    if exclude_self:
        sims[idx] = -1.0
    top_idx = np.argpartition(-sims, range(top_k))[:top_k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(int(i), float(sims[i])) for i in top_idx]
