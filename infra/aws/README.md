# AWS Infrastructure (Terraform)

Provisions a production-shaped deployment of the course recommender on AWS:

```
                    ┌── default ──▶ S3 (static React SPA)
Internet ─https─▶ CloudFront ──┤
                    └── /api/* ──▶ ALB (HTTP :80) ──▶ ECS Fargate (FastAPI, N tasks)
                                                        │  task IAM role
                                                        ├──▶ S3 (model artifacts, read)
                                                        └──▶ Amazon Bedrock (Claude, RAG)
ECR  ◀── container images                             CloudWatch Logs + Container Insights
```

CloudFront serves the whole app **same-origin over HTTPS** — the SPA at `/` and
the API at `/api/*` — so there's no CORS, no mixed content, and no custom domain
or ACM certificate needed (HTTPS comes from `*.cloudfront.net`).

## Resources created

| File | Resources |
|------|-----------|
| `network.tf` | Default VPC/subnet lookups, ALB + service security groups |
| `ecr.tf`     | ECR repository + lifecycle policy |
| `s3.tf`      | Private, encrypted, versioned artifacts bucket |
| `iam.tf`     | ECS execution role + task role (S3 read, Bedrock invoke) |
| `alb.tf`     | Application Load Balancer, target group, listener |
| `ecs.tf`     | ECS cluster, Fargate task definition, service, log group |
| `cdn.tf`     | CloudFront distribution + private S3 site bucket (OAC) + `/api` rewrite function |

## Prerequisites

- Terraform >= 1.5, AWS CLI configured with credentials.
- Bedrock model access enabled for the chosen `llm_model` in your region.

## Deploy

```bash
cd infra/aws
cp terraform.tfvars.example terraform.tfvars   # edit as needed

terraform init
terraform apply                                 # creates ECR, S3, ALB, ECS...

# 1) Build & push the image (or let the GitHub Actions deploy workflow do it):
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin "$(terraform output -raw ecr_repository_url | cut -d/ -f1)"
docker build -f ../../docker/Dockerfile.api -t "$(terraform output -raw ecr_repository_url):latest" ../..
docker push "$(terraform output -raw ecr_repository_url):latest"

# 2) Build artifacts and upload to S3:
cd ../..
ARTIFACT_STORE=s3 \
ARTIFACT_S3_BUCKET="$(cd infra/aws && terraform output -raw artifacts_bucket)" \
python -m src.pipeline --mode build --data data/Coursera.csv

# 3) Roll the service to pick up the image:
aws ecs update-service --force-new-deployment \
  --cluster "$(cd infra/aws && terraform output -raw ecs_cluster_name)" \
  --service "$(cd infra/aws && terraform output -raw ecs_service_name)"

echo "API (origin): $(cd infra/aws && terraform output -raw api_url)"

# 4) Build & publish the frontend behind CloudFront (HTTPS, same-origin /api):
./scripts/deploy_frontend.sh
echo "App (HTTPS): $(cd infra/aws && terraform output -raw web_url)"
```

The public demo link to share is the **`web_url`** (CloudFront) — it serves the
UI and proxies the API. The raw `api_url` (ALB, HTTP) stays available for
Swagger at `/docs` and health checks.

## Cost & teardown

Fargate tasks + ALB accrue hourly cost. Tear everything down with:

```bash
cd infra/aws && terraform destroy
```

## Notes

- Uses the **default VPC** for simplicity. HTTPS is terminated at CloudFront;
  the ALB stays HTTP behind it. For a hardened setup, add a custom domain
  (Route 53 + ACM) on the distribution, restrict the ALB security group to
  CloudFront's managed prefix list, and move tasks into private subnets.
- `bedrock` is the default GenAI provider — no API key needed, the task IAM
  role authorizes `bedrock:InvokeModel`. Switch `llm_provider` to `anthropic`
  and inject `ANTHROPIC_API_KEY` (via SSM/Secrets Manager) to use the Claude API.
