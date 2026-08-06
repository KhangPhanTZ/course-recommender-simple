# Architecture

A content-based course recommender re-shaped as a production, cloud-deployable
service. Three layers — **retrieval engine**, **GenAI/RAG**, **serving** — sit
behind a storage abstraction so the same code runs on a laptop and on AWS.

## System overview

```mermaid
flowchart TB
    subgraph Client
        UI["Streamlit UI"]
        Curl["HTTP clients / curl"]
    end

    subgraph Serving["FastAPI service"]
        R["/recommend, /chat, /roadmap, /similar, /courses, /health"]
        DEP["cached singletons\n(Recommender + LLM)"]
    end

    subgraph Engine["Retrieval engine (src/recsys)"]
        ENC["Query encoder\n(TF-IDF / SBERT)"]
        IDX["ANN index\n(FAISS / numpy fallback)"]
        RR["Cross-encoder rerank\n(optional)"]
    end

    subgraph GenAI["GenAI / RAG (src/llm)"]
        QU["Query understanding\n(NL -> filters)"]
        EX["Explanations (RAG)"]
        PROV["Provider abstraction"]
    end

    subgraph Storage["Artifact storage (src/storage)"]
        LOC["Local FS"]
        S3["AWS S3"]
    end

    UI --> R
    Curl --> R
    R --> DEP --> ENC --> IDX --> RR --> R
    DEP --> QU --> ENC
    DEP --> EX
    QU --> PROV
    EX --> PROV
    PROV -->|anthropic| CLAUDE["Claude API"]
    PROV -->|bedrock| BR["Amazon Bedrock"]
    DEP --> Storage
    Engine --> Storage
```

## Request flow: `POST /recommend`

```mermaid
sequenceDiagram
    participant U as Client
    participant API as FastAPI
    participant LLM as GenAI layer
    participant REC as Recommender
    participant IDX as ANN index

    U->>API: {query, top_k, explain}
    API->>LLM: understand_query(query)
    LLM-->>API: {search, level, category}   %% LLM or heuristic fallback
    API->>REC: recommend(search, filters)
    REC->>IDX: nearest neighbours (cosine)
    IDX-->>REC: candidate ids + scores
    REC->>REC: apply filters (+ optional rerank)
    REC-->>API: top-k courses
    opt explain=true
        API->>LLM: explain_recommendations(query, courses)
        LLM-->>API: grounded rationale (RAG)
    end
    API-->>U: results + filters + explanation
```

## Two-stage retrieval

1. **Recall** — the query is embedded and searched against the course index.
   - `sbert` backend: dense Sentence-BERT vectors, FAISS inner-product on
     normalized vectors (= cosine). Falls back to a vectorized numpy search
     when FAISS is unavailable, so behaviour is identical, only slower.
   - `tfidf` backend: sparse TF-IDF + exact cosine. No model downloads.
2. **Precision** — an optional cross-encoder rescoring of the shortlist
   (`ENABLE_RERANK=true`), blending first-stage and cross-encoder scores.

Filters (level/category) are applied as a post-retrieval mask so semantic
ranking is preserved within the filtered set.

## GenAI / RAG layer

- **Query understanding** rewrites a messy request into a clean semantic query
  plus structured filters. An LLM does this when configured; otherwise a
  deterministic heuristic keeps the endpoint fully functional.
- **Explanations** are Retrieval-Augmented: the *retrieved* courses are the
  grounding context, so the model explains real results rather than
  hallucinating a catalog.
- **Provider abstraction** makes `anthropic` (Claude API) and `bedrock`
  (Claude on AWS Bedrock) interchangeable via one env var. On ECS, Bedrock
  needs no API key — the task IAM role authorizes it.

## Storage abstraction

`ArtifactStore` (local / S3) is the seam that decouples training from serving.
`python -m src.pipeline --mode build` writes artifacts through it; the service
reads through it. Switching from a laptop to AWS is a change of
`ARTIFACT_STORE=s3` + `ARTIFACT_S3_BUCKET`, no code change.

Artifacts: `meta.json`, `courses.parquet`, and either
(`embeddings.npy` + `vector.index`) or (`tfidf_vectorizer.pkl` + `X_tfidf.npz`),
plus `kmeans.pkl` for the cluster labels.

## AWS deployment

```mermaid
flowchart LR
    Dev["Developer / CI"] -->|docker push| ECR
    Dev -->|terraform apply| TF["Terraform"]
    GHA["GitHub Actions\n(deploy-aws)"] -->|OIDC| AWS
    subgraph AWS
        ALB["ALB :80"] --> ECS["ECS Fargate\nFastAPI tasks"]
        ECS -->|task role| S3B["S3 artifacts"]
        ECS -->|task role| BRK["Bedrock (Claude)"]
        ECS --> CW["CloudWatch Logs"]
        ECR["ECR"] --> ECS
    end
    Users --> ALB
```

See [`infra/aws/README.md`](../infra/aws/README.md) for the deploy runbook.
