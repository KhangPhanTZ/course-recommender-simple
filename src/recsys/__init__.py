"""Core recommendation engine package.

Modules:
- ``features``:   text cleaning and corpus construction
- ``embeddings``: TF-IDF and Sentence-BERT vectorizers
- ``index``:      approximate nearest-neighbour search (FAISS + numpy fallback)
- ``rerank``:     cross-encoder reranking
- ``recommender``: high-level service that ties artifacts together for serving
"""
