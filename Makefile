.PHONY: help install dev build query api web web-install web-build test lint fmt docker-build up down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime deps (full/local)
	pip install -r requirements.txt

dev: ## Install lightweight CI/test deps + linters
	pip install -r requirements-ci.txt ruff

build: ## Build artifacts (vectors, FAISS index, clusters)
	python -m src.pipeline --mode build --data data/Coursera.csv

query: ## Query from the CLI: make query Q="deep learning with pytorch"
	python -m src.pipeline --mode query --text "$(Q)"

api: ## Run the FastAPI service locally on :8000
	uvicorn src.api.main:app --reload --port 8000

web-install: ## Install frontend dependencies
	cd frontend && npm install

web: ## Run the React/Vite dev server on :5173 (proxies /api -> :8000)
	cd frontend && npm run dev

web-build: ## Build the frontend for production
	cd frontend && npm run build

test: ## Run unit tests (no network, no LLM calls)
	pytest

test-llm: ## Run live LLM tests against the configured provider (costs tokens)
	RUN_LLM_TESTS=1 pytest tests/test_llm_live.py -v

smoke-llm: ## Smoke-test the GenAI layer and report model vs fallback per stage
	python scripts/smoke_llm.py

lint: ## Lint with ruff
	ruff check src tests

fmt: ## Auto-fix lint issues
	ruff check --fix src tests

docker-build: ## Build the API + web images
	docker build -f docker/Dockerfile.api -t course-recommender-api:local .
	docker build -f frontend/Dockerfile -t course-recommender-web:local frontend

up: ## Start the full stack (API + web) with docker compose
	docker compose up --build

down: ## Stop the stack
	docker compose down

clean: ## Remove caches and local artifacts
	rm -rf .pytest_cache __pycache__ **/__pycache__ artifacts/*.npy artifacts/*.pkl artifacts/*.json artifacts/*.parquet artifacts/*.index artifacts/*.npz
