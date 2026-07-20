"""Streamlit demo UI for the course recommender.

Thin client: search & similar go through the FastAPI service (so the demo
exercises the same code path as production). The clusters map reads the local
artifacts directory directly when available.

Configure the backend API with the ``RECSYS_API_URL`` env var
(default ``http://localhost:8000``).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("RECSYS_API_URL", "http://localhost:8000").rstrip("/")
ART = Path(os.getenv("ARTIFACT_DIR", "artifacts"))

st.set_page_config(page_title="Course Recommender", layout="wide")


@st.cache_data(show_spinner=False)
def load_courses():
    path = ART / "courses.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_embedding():
    path = ART / "umap_embedding.npy"
    return np.load(path) if path.exists() else None


def api_get(path: str):
    try:
        return requests.get(f"{API_URL}{path}", timeout=30).json()
    except requests.RequestException as exc:
        st.error(f"API unreachable at {API_URL}: {exc}")
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


st.title("🎓 Content-Based Course Recommender")
health = api_get("/health") or {}
cols = st.columns(4)
cols[0].metric("API status", health.get("status", "unknown"))
cols[1].metric("Backend", health.get("backend", "—"))
cols[2].metric("Courses", health.get("n_courses", "—"))
cols[3].metric("LLM", health.get("llm_provider") or "off")

tab1, tab2, tab3 = st.tabs(["🔎 Search (RAG)", "🎯 Similar by Course", "🗺️ Clusters Map"])

with tab1:
    q = st.text_input("Describe what you want to learn:", "deep learning with python for beginners")
    c1, c2, c3 = st.columns(3)
    topk = c1.slider("Top-K", 5, 30, 10)
    explain = c2.checkbox("LLM explanation (RAG)", value=True)
    rerank = c3.checkbox("Cross-encoder rerank", value=False)
    if st.button("Search", type="primary"):
        data = api_post("/recommend", {
            "query": q, "top_k": topk, "explain": explain,
            "understand": True, "rerank": rerank,
        })
        if data:
            if data.get("filters"):
                st.caption(f"Understood filters: `{data['filters']}` · resolved query: _{data['resolved_query']}_")
            if data.get("explanation"):
                st.info(data["explanation"])
            st.dataframe(pd.DataFrame(data["results"]).reset_index(drop=True), use_container_width=True)

with tab2:
    df = load_courses()
    if df is None:
        st.info("Build artifacts first: `python -m src.pipeline --mode build`.")
    else:
        topk2 = st.slider("Top-K similar", 5, 30, 10, key="topk2")
        titles = df["title"].astype(str).tolist()
        sel = st.selectbox("Pick a course", options=range(len(titles)), format_func=lambda i: titles[i][:120])
        if st.button("Find similar"):
            course_id = df.iloc[sel].get("id", sel)
            data = api_post("/similar", {"course_id": int(course_id), "top_k": topk2})
            if data is not None:
                st.dataframe(pd.DataFrame(data).reset_index(drop=True), use_container_width=True)

with tab3:
    df = load_courses()
    emb = load_embedding()
    if df is None or emb is None:
        st.info("Run build with `compute_viz: true` to see the clusters map.")
    else:
        import plotly.express as px

        plot_df = pd.DataFrame(emb, columns=["x", "y"])
        plot_df["title"] = df["title"].astype(str)
        plot_df["cluster"] = df["cluster"].astype(str) if "cluster" in df else "0"
        fig = px.scatter(plot_df, x="x", y="y", color="cluster", hover_data=["title"], title="UMAP of Courses")
        st.plotly_chart(fig, use_container_width=True)
