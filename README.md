# Course Recommender

A content-based course recommendation system built as a production service rather
than a notebook. It pairs a two-stage retriever (ANN + optional cross-encoder)
with a GenAI layer for query understanding and grounded explanations, serves both
through FastAPI, and ships with Docker and Terraform for an AWS ECS/Fargate
deployment.

Built to demonstrate the end-to-end skill set for an AI Engineer role: retrieval,
LLM integration, API design, containerization, IaC, and CI/CD.

## Architecture

```
Client -> FastAPI -> Retrieval engine (encode -> ANN -> rerank)
             |            `-> Artifact store (local | S3)
             `-> GenAI layer (query understanding + explanations)
                      `-> Claude API | Amazon Bedrock

Deploy:  ECR -> ECS Fargate (behind ALB) -> S3 + Bedrock + CloudWatch
```

Full diagrams: [docs/architecture.md](docs/architecture.md)

## Features

- Two-stage retrieval: Sentence-BERT + FAISS ANN, with a numpy fallback when
  FAISS is unavailable, plus optional cross-encoder reranking.
- GenAI layer: natural-language query understanding (text to filters) and
  retrieval-grounded explanations. Works with the Claude API or Amazon Bedrock
  behind one interface, and degrades to deterministic templates when no LLM is
  configured.
- KMeans clustering with a UMAP projection for catalog exploration.
- FastAPI service with health probes, request timing, and OpenAPI docs.
- React + Vite + Tailwind web UI: landing page, live search, explore, and
  architecture pages, dark/light themed.
- Pluggable artifact store: identical code paths for local disk and S3.
- Docker Compose for local full-stack, Terraform for AWS, GitHub Actions for CI/CD.
- 31 unit tests and ruff lint, green in CI.

## Project structure

```
src/
  recsys/       retrieval engine: index (FAISS/numpy), recommender, rerank
  llm/          GenAI layer: provider abstraction (Claude/Bedrock) + service
  storage/      artifact store: local filesystem + S3
  api/          FastAPI app: routes, schemas, dependency injection
  models/       TF-IDF / SBERT vectorizers, KMeans, UMAP
  pipeline.py   end-to-end build
  eval.py       offline evaluation (precision/recall/NDCG, silhouette)
  settings.py   env-driven runtime config
frontend/       React + Vite + Tailwind SPA
docker/         Dockerfile.api
infra/aws/      Terraform: ECR, ECS, ALB, S3, IAM, CloudWatch
.github/        CI, AWS deploy, and Terraform workflows
tests/          pytest suite
```

## Quickstart

Requires Python 3.11+ and Node 20+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

### 1. Get the dataset

Kaggle: [Multi-Platform Online Courses Dataset](https://www.kaggle.com/datasets/everydaycodings/multi-platform-online-courses-dataset).
Only the Coursera slice is used. Place it at `data/Coursera.csv` (git-ignored) and
adjust the column mapping in `config/config.yaml` if your columns differ.

```bash
pip install kaggle
kaggle datasets download -d everydaycodings/multi-platform-online-courses-dataset -p data --unzip
```

### 2. Build artifacts

```bash
python -m src.pipeline --mode build --data data/Coursera.csv
```

First run downloads the Sentence-BERT model (~90 MB). To skip it entirely, set
`use_sbert: false` in `config/config.yaml` to use TF-IDF instead.

### 3. Run

Query from the CLI:

```bash
python -m src.pipeline --mode query --text "deep learning with pytorch for beginners"
```

Or run the API and web UI in two terminals:

```bash
uvicorn src.api.main:app --reload --port 8000    # http://localhost:8000/docs
cd frontend && npm install && npm run dev        # http://localhost:5173
```

The Vite dev server proxies `/api` to `http://localhost:8000`, so the UI and API
share an origin.

Docker Compose runs both (web on `:8080`, API on `:8000`):

```bash
docker compose up --build
```

## API

```bash
curl -s localhost:8000/recommend -H 'content-type: application/json' -d '{
  "query": "I want to learn deep learning with pytorch as a beginner",
  "top_k": 5, "explain": true
}'
```

The response contains the ranked courses, the filters the system inferred from
the query (`{"level": "beginner"}`), and an explanation grounded in the retrieved
results.

| Method | Path                 | Description                              |
|--------|----------------------|------------------------------------------|
| GET    | `/health`            | Liveness probe, backend and catalog size |
| POST   | `/recommend`         | Rank courses for a natural-language goal |
| POST   | `/similar`           | Courses similar to a given course id     |
| GET    | `/courses/{id}`      | Single course record                     |
| GET    | `/map`               | 2D cluster projection for the map view   |

## Configuration

Two layers:

- `config/config.yaml` — dataset column mapping, embedding backend (`use_sbert`),
  clustering, retrieval parameters.
- Environment variables (`.env.example`) — artifact storage, LLM provider and
  model, reranking. Documented in [src/settings.py](src/settings.py).

Switch retrieval backends with a single flag: `use_sbert: true` for semantic
search with FAISS, or `false` for TF-IDF with no model downloads (used by CI).

To enable the GenAI layer, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`
in `.env`. Leave it at `disabled` to run without an API key. Note that the
parser in `src/settings.py` does not strip trailing `#` comments, so keep
comments on their own lines in `.env`.

## Evaluation

Offline metrics against category and skill labels, plus clustering quality:

```bash
python -m src.eval --metric category --topk 10
python -m src.eval --metric skills   --topk 10
python -m src.eval --metric silhouette
```

## Deploy to AWS

Terraform provisions ECR, an S3 artifact bucket, an ECS Fargate service behind an
ALB, IAM roles (S3 read, Bedrock invoke), and CloudWatch logging. The GenAI layer
defaults to Bedrock in the cloud, so no API key is needed — the task role
authorizes `bedrock:InvokeModel`.

```bash
cd infra/aws
cp terraform.tfvars.example terraform.tfvars
terraform init && terraform apply
```

Then push the image and upload artifacts. Full runbook:
[infra/aws/README.md](infra/aws/README.md).

Note that the Fargate tasks and the ALB accrue hourly cost. Tear everything down
with `terraform destroy` when finished.

## Development

```bash
make dev      # install lint and test dependencies
make test     # pytest
make lint     # ruff
```
