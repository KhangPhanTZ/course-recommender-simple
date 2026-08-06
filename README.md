# Pathfinder — AI Course Recommender

A content-based course recommender: a **two-stage retriever** (ANN + optional
cross-encoder) paired with a **GenAI/RAG** layer for query understanding and
grounded explanations. FastAPI backend, React + Tailwind UI, deployable to
Render (one click) or AWS (Terraform).

**🔗 Live demo:** https://course-recommender-6y4z.onrender.com
_(free tier — the first request may take ~50s while the instance wakes up)_

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/KhangPhanTZ/course-recommender-simple)

## Features

- **Multi-platform catalog** — Coursera, Udemy, and edX datasets normalized onto one schema and merged; drop a CSV into `data/` and the build auto-detects it (see [`data/README.md`](data/README.md)).
- **Two-stage retrieval** — Sentence-BERT + FAISS (numpy fallback) + optional cross-encoder rerank.
- **GenAI / RAG** — natural-language query understanding and retrieval-grounded explanations, via Claude API or AWS Bedrock, with deterministic fallbacks when no LLM is set.
- **Chat advisor** — a conversational RAG assistant that explains syllabi and learning paths, grounded in the catalog.
- **Web UI** — React + Vite + Tailwind SPA (landing, search, chat, about), dark/light.
- **Production-shaped** — pluggable local⇄S3 storage, Docker, Terraform (AWS ECS/Fargate), CI/CD, tests.

**Stack:** Python · FastAPI · scikit-learn · Sentence-BERT · FAISS · React · Vite · Tailwind · Docker · Terraform · AWS · GitHub Actions

## Architecture

```
Client → FastAPI → retrieval engine (encode → ANN → rerank)
             │            └─ artifact store (local | S3)
             └─ GenAI/RAG (understand + explain) → Claude API | Bedrock
```

Diagrams and details: [docs/architecture.md](docs/architecture.md).

## Quickstart

Requires Python 3.11+ and Node 20+.

```bash
pip install -r requirements.txt
python -m src.pipeline --mode build --data data/    # merge every dataset in data/
uvicorn src.api.main:app --port 8000                # API + docs at /docs
cd frontend && npm install && npm run dev           # UI at :5173
```

Data: point `--data` at a **directory** to merge every recognized Coursera / Udemy / edX file, or at a single CSV. When `data/` is empty the demo build falls back to the bundled synthetic multi-platform sample (`examples/catalog/`). See [`data/README.md`](data/README.md) for the supported Kaggle datasets and how to add them. Set `use_sbert: false` in `config/config.yaml` for a no-download TF-IDF setup. To enable the GenAI layer, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `.env`.

## Deploy

- **Render (one click):** the button above, driven by [`render.yaml`](render.yaml).
- **AWS:** `./scripts/deploy_aws.sh` provisions ECR, ECS Fargate, ALB, S3, IAM and Bedrock via Terraform ([infra/aws](infra/aws)).

## Development

```bash
make test    # pytest (hermetic: no network, no LLM calls)
make lint    # ruff
```
