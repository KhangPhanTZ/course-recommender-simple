import sys
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

st.set_page_config(page_title="Course Recommender", layout="wide")
ART = Path("artifacts")

@st.cache_data
def load_courses():
    path = ART / "courses.parquet"
    if not path.exists():
        st.error("Artifacts not found. Please run: python -m src.pipeline --mode build")
        st.stop()
    return pd.read_parquet(path)

@st.cache_data
def load_embedding():
    emb_path = ART / "umap_embedding.npy"
    if emb_path.exists():
        return np.load(emb_path)
    return None

def query_free_text(q: str, topk: int = 10):
    import sys
    from pathlib import Path
    sys.path.append(str(Path(".").resolve()))
    from src.pipeline import query_similar
    df = query_similar("config/config.yaml", q, topk)
    cols = ["id","title","category","level","rating","score"]
    return df[[c for c in cols if c in df.columns]]


st.title("🎓 Content-Based Course Recommender")
st.write("Search similar courses by **free text** or **pick an existing course**.")

tab1, tab2, tab3 = st.tabs(["🔎 Search", "🎯 Similar by Course", "🗺️ Clusters Map"])

with tab1:
    q = st.text_input("Describe what you want to learn:", "deep learning with python for beginners")
    topk = st.slider("Top-K results", 5, 30, 10)
    if st.button("Search"):
        st.write("**Results (terminal-style preview)**")
        res = query_free_text(q, topk)
        st.dataframe(res.reset_index(drop=True))
        
with tab2:
    df = load_courses()
    topk2 = st.slider("Top-K similar", 5, 30, 10, key="topk2")
    titles = df["title"].astype(str).fillna("").tolist()
    sel = st.selectbox("Pick a course", options=range(len(titles)), format_func=lambda i: titles[i][:120])
    # Compute similarity in-app
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Simple TF-IDF on the fly over cleaned text fields (for demo)
    corpus = (df["title"].fillna("") + " | " + df["description"].fillna("") + " | " + df["skills"].fillna("")).str.lower().tolist()
    vec = TfidfVectorizer(max_features=40000, ngram_range=(1,2))
    X = vec.fit_transform(corpus)
    sims = cosine_similarity(X[sel], X).ravel()
    sims[sel] = -1.0
    idx = np.argpartition(-sims, range(topk2))[:topk2]
    idx = idx[np.argsort(-sims[idx])]
    st.subheader("Top similar")
    st.dataframe(df.iloc[idx][["title","category","level","rating","url"]].assign(score=sims[idx]).reset_index(drop=True))

with tab3:
    df = load_courses()
    emb = load_embedding()
    if emb is None:
        st.info("Run build first to compute UMAP embedding.")
    else:
        import plotly.express as px
        plot_df = pd.DataFrame(emb, columns=["x","y"])
        plot_df["title"] = df["title"].astype(str)
        plot_df["cluster"] = df["cluster"].astype(str)
        fig = px.scatter(plot_df, x="x", y="y", color="cluster", hover_data=["title"], title="UMAP of Courses")
        st.plotly_chart(fig, use_container_width=True)
