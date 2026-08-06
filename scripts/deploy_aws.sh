#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command AWS deploy for the Course Recommender API.
#
# It: applies the Terraform stack (ECR, S3, ALB, ECS Fargate, IAM, logs),
# builds & pushes the API image, builds the model artifacts straight into S3,
# rolls the ECS service, and prints the public API URL.
#
# Prerequisites (run these once, on YOUR machine — this script needs your
# authenticated AWS session; it cannot run from a sandbox):
#   - aws CLI configured:      aws configure         (or SSO / env creds)
#   - docker running
#   - terraform >= 1.5
#   - a dataset at data/Coursera.csv  (see README "Get the dataset")
#   - Amazon Bedrock model access enabled in your region (for the GenAI layer)
#
# Usage:
#   ./scripts/deploy_aws.sh            # deploy
#   DATA=data/Coursera.csv ./scripts/deploy_aws.sh
#
# Tear down later with:  cd infra/aws && terraform destroy
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="$REPO_ROOT/infra/aws"
DATA="${DATA:-data/Coursera.csv}"

cyan() { printf "\033[36m==> %s\033[0m\n" "$1"; }
die()  { printf "\033[31mERROR: %s\033[0m\n" "$1" >&2; exit 1; }

# --- 0. preconditions ------------------------------------------------------
command -v aws >/dev/null       || die "aws CLI not found."
command -v docker >/dev/null    || die "docker not found."
command -v terraform >/dev/null || die "terraform not found."
aws sts get-caller-identity >/dev/null 2>&1 || die "AWS credentials not configured (run 'aws configure')."
[ -f "$REPO_ROOT/$DATA" ] || die "Dataset not found at $DATA (see README)."

# --- 1. provision infrastructure ------------------------------------------
cyan "Applying Terraform (ECR, S3, ALB, ECS, IAM)..."
cd "$INFRA_DIR"
terraform init -input=false
terraform apply -auto-approve

REGION="$(terraform output -raw aws_region 2>/dev/null || echo "${AWS_REGION:-us-east-1}")"
ECR_URL="$(terraform output -raw ecr_repository_url)"
BUCKET="$(terraform output -raw artifacts_bucket)"
CLUSTER="$(terraform output -raw ecs_cluster_name)"
SERVICE="$(terraform output -raw ecs_service_name)"
API_URL="$(terraform output -raw api_url)"
REGISTRY="${ECR_URL%%/*}"

# --- 2. build & push the API image ----------------------------------------
cyan "Logging in to ECR and pushing the API image..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker build -f "$REPO_ROOT/docker/Dockerfile.api" -t "$ECR_URL:latest" "$REPO_ROOT"
docker push "$ECR_URL:latest"

# --- 3. build model artifacts straight into S3 -----------------------------
cyan "Building artifacts into s3://$BUCKET ..."
( cd "$REPO_ROOT" && \
  ARTIFACT_STORE=s3 ARTIFACT_S3_BUCKET="$BUCKET" ARTIFACT_S3_PREFIX=artifacts AWS_REGION="$REGION" \
  python -m src.pipeline --mode build --data "$DATA" )

# --- 4. roll the service ---------------------------------------------------
cyan "Rolling the ECS service..."
aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
  --force-new-deployment --region "$REGION" >/dev/null
cyan "Waiting for the service to stabilize (a few minutes)..."
aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION"

# --- 5. done ---------------------------------------------------------------
printf "\n\033[32m✅ Deployed.\033[0m\n"
echo "   API:   $API_URL"
echo "   Docs:  $API_URL/docs"
echo "   Health:$API_URL/health"
echo
echo "Now publish the web UI over HTTPS (CloudFront, same-origin /api):"
echo "   ./scripts/deploy_frontend.sh        # prints the shareable web_url"
