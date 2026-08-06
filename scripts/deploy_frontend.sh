#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build the React SPA and publish it to the CloudFront-fronted S3 bucket.
#
# The app is served same-origin: the SPA lives at / and calls the API at /api
# (CloudFront proxies /api/* to the ALB). So no CORS, no mixed content, HTTPS
# for free via *.cloudfront.net.
#
# Prerequisites (run on YOUR machine / CloudShell, with AWS creds):
#   - the Terraform stack applied (infra/aws) incl. cdn.tf
#   - aws CLI + node/npm
#
# Usage:  ./scripts/deploy_frontend.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="$REPO_ROOT/infra/aws"

cyan() { printf "\033[36m==> %s\033[0m\n" "$1"; }
die()  { printf "\033[31mERROR: %s\033[0m\n" "$1" >&2; exit 1; }

command -v aws >/dev/null || die "aws CLI not found."
command -v npm >/dev/null || die "npm not found."

cyan "Reading Terraform outputs..."
cd "$INFRA_DIR"
BUCKET="$(terraform output -raw web_bucket)"       || die "No web_bucket output — apply cdn.tf first."
DIST_ID="$(terraform output -raw cloudfront_distribution_id)"
WEB_URL="$(terraform output -raw web_url)"
REGION="$(terraform output -raw aws_region 2>/dev/null || echo us-east-1)"

cyan "Building the SPA (same-origin API at /api)..."
cd "$REPO_ROOT/frontend"
npm ci
VITE_API_URL=/api npm run build   # default is already /api; set explicitly for clarity

cyan "Uploading to s3://$BUCKET ..."
aws s3 sync dist/ "s3://$BUCKET/" --delete --region "$REGION"

cyan "Invalidating CloudFront cache..."
aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" >/dev/null

printf "\n\033[32m✅ Frontend live:\033[0m %s\n" "$WEB_URL"
echo "   (CloudFront can take a few minutes to propagate the first time.)"
