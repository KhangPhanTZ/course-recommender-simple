# Pathfinder — AI Course Recommender

A content-based course recommender: a **two-stage retriever** (ANN + optional
cross-encoder) paired with a **GenAI/RAG** layer for query understanding and
grounded explanations. FastAPI backend, React + Tailwind UI, deployable to
Render (one click) or AWS (Terraform).

**🔗 Live demo:** https://course-recommender-6y4z.onrender.com
_(free tier — the first request may take ~50s while the instance wakes up)_

What the demo runs, so the numbers are read for what they are: the real Coursera
catalog (~1,100 courses) on the **TF-IDF** backend rather than Sentence-BERT, to
fit the free tier — so the Metrics figures reflect keyword retrieval, and a
Sentence-BERT build scores differently. Udemy and edX are supported but not
committed here, so the demo is single-platform. `/api/health` reports whether the
GenAI layer is actually reachable; when it is not, the service answers from
deterministic templates instead of failing.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/KhangPhanTZ/course-recommender-simple)

## Features

- **Multi-platform catalog** — Coursera, Udemy, and edX datasets normalized onto one schema and merged; drop a CSV into `data/` and the build auto-detects it (see [`data/README.md`](data/README.md)).
- **Two-stage retrieval** — Sentence-BERT + FAISS (numpy fallback) + optional cross-encoder rerank.
- **GenAI / RAG** — natural-language query understanding and retrieval-grounded explanations, via Claude API or AWS Bedrock, with deterministic fallbacks when no LLM is set.
- **Chat advisor** — a conversational RAG assistant that explains syllabi and learning paths, grounded in the catalog. Every course it (or a roadmap) surfaces is clickable for a summary, skills, and a link to the real course.
- **Catalog browse** — a `/courses` page and endpoint to search/filter the whole catalog (title/skills, source, level) with pagination.
- **Career-track roadmaps** — pick a track (Data Analyst, ML Engineer, MLOps, PM, QA…) and get a tiered Foundation→Specialization roadmap, each tier grounded in real catalog courses, with bridges to adjacent tracks.
- **Evaluation dashboard** — offline retrieval metrics (Precision/Recall/MRR/nDCG@k, hit-rate), query latency (p50/p95), and clustering silhouette, surfaced at `/metrics` and a Metrics page.
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
- **AWS:** Terraform ([infra/aws](infra/aws)) provisions ECR, ECS Fargate, ALB, S3, IAM, Bedrock, and a **CloudFront** distribution that serves the SPA over **HTTPS** and proxies `/api/*` to the ALB (same-origin, no custom domain needed). Deploy the API with `./scripts/deploy_aws.sh`, then publish the UI with `./scripts/deploy_frontend.sh`; the shareable link is the `web_url` output.
- **From CI:** [`deploy-aws.yml`](.github/workflows/deploy-aws.yml) does the same from GitHub Actions, authenticating through OIDC against the role in [`github_oidc.tf`](infra/aws/github_oidc.tf) — no long-lived AWS key in repository secrets, and the multi-gigabyte image push runs on GitHub's network rather than a laptop's uplink. Set `github_repository` in `terraform.tfvars`, apply, then set the repository variables the workflow reads.

### Deployment status

The AWS path has been applied and exercised end to end, not just validated:
ECS ran the pushed image, the ALB served `/health` and `/recommend`, CloudFront
served the SPA over HTTPS, and the GenAI layer answered **through Bedrock**.

Two things that path taught, both fixed here: current Claude models reject
on-demand invocation of a bare foundation-model id and must be called through an
**inference profile**, so the task policy grants the profile plus the foundation
models it routes to across regions; and the AWS deployment builds its artifacts
with the **TF-IDF** backend, so semantic matching on non-English queries is
weaker there than a Sentence-BERT build would be.

Tear the stack down with `terraform destroy` when you are done — the ALB and the
CloudFront distribution bill by the hour whether or not anything is serving.

## Development

```bash
make test    # pytest (hermetic: no network, no LLM calls)
make lint    # ruff
```
