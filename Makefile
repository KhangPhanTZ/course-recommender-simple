.PHONY: help install dev build query api ui test lint fmt docker-build compose up down clean

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

ui: ## Run the Streamlit demo on :8501
	streamlit run app/streamlit_app.py

test: ## Run unit tests
	pytest

lint: ## Lint with ruff
	ruff check src app tests

fmt: ## Auto-fix lint issues
	ruff check --fix src app tests

docker-build: ## Build the API image
	docker build -f docker/Dockerfile.api -t course-recommender-api:local .

up: ## Start the full stack (API + UI) with docker compose
	docker compose up --build

down: ## Stop the stack
	docker compose down

clean: ## Remove caches and local artifacts
	rm -rf .pytest_cache __pycache__ **/__pycache__ artifacts/*.npy artifacts/*.pkl artifacts/*.json artifacts/*.parquet artifacts/*.index artifacts/*.npz
