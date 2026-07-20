# Course Recommender — Retrieval + GenAI, deployable on AWS

A content-based course recommendation system, engineered like a production
service rather than a notebook. It pairs a **two-stage retriever** (ANN +
cross-encoder) with a **GenAI/RAG layer** (query understanding + grounded
explanations), exposes everything through a **FastAPI** service, and ships with
**Docker** and **Terraform** for a one-command **AWS ECS/Fargate** deployment.

> Built to demonstrate the end-to-end skill set for an **AI Engineer** role:
> retrieval, LLM integration, API design, containerization, IaC, and CI/CD.

## Highlights

- 🔎 **Two-stage retrieval** — Sentence-BERT + FAISS ANN (numpy fallback), with
  optional cross-encoder reranking for precision.
- 🧠 **GenAI / RAG** — LLM query understanding (NL → filters) and
  retrieval-grounded "why these courses" explanations. Works with the
  **Claude API** or **Amazon Bedrock** behind one interface; degrades to
  deterministic fallbacks when no LLM is configured.
- 🧩 **Clustering + UMAP** map for catalog exploration.
- ⚡ **FastAPI** service with health probes, request timing, and OpenAPI docs.
- ☁️ **Cloud-native storage** — pluggable local ⇄ S3 artifact store; identical
  code on a laptop and on AWS.
- 🐳 **Docker + compose**, 🏗️ **Terraform** (ECR, ECS Fargate, ALB, S3, IAM,
  Bedrock), and 🔁 **GitHub Actions** CI/CD (test, lint, image build, deploy).
- ✅ **29 unit tests** + ruff lint, all green in CI.

## Architecture

See **[docs/architecture.md](docs/architecture.md)** for full diagrams.

```
Client ─▶ FastAPI ─▶ Retrieval engine (encode ▶ ANN ▶ rerank)
              │            └─▶ Artifact store (local | S3)
              └─▶ GenAI/RAG (query understanding + explanations)
                       └─▶ Claude API | Amazon Bedrock
Deploy:  ECR ▶ ECS Fargate (behind ALB) ▶ S3 + Bedrock + CloudWatch
```

## Project structure

```
src/
├── recsys/          # retrieval engine: index (FAISS/numpy), recommender, rerank
├── llm/             # GenAI/RAG: provider abstraction (Claude/Bedrock) + service
├── storage/         # artifact store: local filesystem + S3
├── api/             # FastAPI app: routes, schemas, DI
├── models/          # TF-IDF / SBERT vectorizers, KMeans, UMAP
├── pipeline.py      # end-to-end build
├── eval.py          # offline evaluation (precision/recall/NDCG, silhouette)
└── settings.py      # env-driven runtime config
app/streamlit_app.py # demo UI (thin client over the API)
docker/              # Dockerfiles (API, Streamlit) + compose
infra/aws/           # Terraform: ECR, ECS, ALB, S3, IAM, CloudWatch
.github/workflows/   # ci.yml, deploy-aws.yml, terraform.yml
tests/               # pytest suite
```

## Quickstart (local)

```bash
pip install -r requirements.txt          # or: make install
cp .env.example .env                      # optional: add ANTHROPIC_API_KEY

# 1) Build artifacts from your dataset
python -m src.pipeline --mode build --data data/Coursera.csv

# 2a) Query from the CLI
python -m src.pipeline --mode query --text "deep learning with pytorch for beginners"

# 2b) Or run the API + UI
uvicorn src.api.main:app --reload --port 8000   # http://localhost:8000/docs
streamlit run app/streamlit_app.py              # http://localhost:8501
```

Prefer Docker? `docker compose up --build` starts the API and UI together.

### Example API call

```bash
curl -s localhost:8000/recommend -H 'content-type: application/json' -d '{
  "query": "I want to learn deep learning with pytorch as a beginner",
  "top_k": 5, "explain": true
}' | jq
```

Response includes the ranked courses, the **filters the system inferred**
(`{"level": "beginner"}`), and an LLM-generated **explanation** grounded in the
retrieved results.

## Configuration

Two layers:

- `config/config.yaml` — dataset column mapping, backend (`use_sbert`),
  clustering, retrieval params.
- Environment variables (see `.env.example`) — storage, LLM provider/model,
  reranking. Documented in [`src/settings.py`](src/settings.py).

Switch backends with a single flag: `use_sbert: true` (semantic, FAISS) or
`false` (TF-IDF, zero downloads — used by CI).

## Deploy to AWS

Terraform provisions ECR, an S3 artifact bucket, an ECS Fargate service behind
an ALB, IAM roles (S3 read + Bedrock invoke), and CloudWatch logging. The
GenAI layer defaults to **Bedrock**, so no API key is needed in the cloud — the
task IAM role authorizes `bedrock:InvokeModel`.

```bash
cd infra/aws && cp terraform.tfvars.example terraform.tfvars
terraform init && terraform apply
# then push the image + upload artifacts (see infra/aws/README.md)
```

Full runbook: **[infra/aws/README.md](infra/aws/README.md)**.

## Evaluation

Offline metrics against category/skill labels and clustering quality:

```bash
python -m src.eval --metric category --topk 10
python -m src.eval --metric skills   --topk 10
python -m src.eval --metric silhouette
```

## Development

```bash
make dev      # install lint/test deps
make test     # pytest (29 tests)
make lint     # ruff
```

## Dataset

- Kaggle: *Multi-Platform Online Courses Dataset* —
  https://www.kaggle.com/datasets/everydaycodings/multi-platform-online-courses-dataset
- Only the Coursera slice is used. Place it at `data/Coursera.csv`
  (git-ignored) and adjust `config/config.yaml` if your columns differ.
