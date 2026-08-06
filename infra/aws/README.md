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
| `autoscaling.tf` | Application Auto Scaling target + CPU target-tracking policy |
| `alarms.tf`  | CloudWatch alarms (CPU, target 5xx, unhealthy hosts) + optional SNS email |

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

## Scaling & observability

The ECS service **autoscales** on average CPU between `min_capacity` and
`max_capacity` (target `cpu_target`%). Terraform ignores `desired_count` drift so
it doesn't fight the scaler. **CloudWatch alarms** watch CPU saturation, target
5xx errors, and unhealthy hosts; set `alarm_email` to get SNS notifications
(confirm the subscription email AWS sends), or leave it empty for console-only.

```hcl
# terraform.tfvars
min_capacity = 1     # scale to a single task when idle (cheapest)
max_capacity = 4
cpu_target   = 60
alarm_email  = "you@example.com"   # optional
```

## CI-driven deploys (optional)

`deploy-aws.yml` builds/pushes the image and rolls ECS on every push to `main`.
Two **opt-in** `workflow_dispatch` inputs extend it so data/UI updates flow
through CI too:

- `rebuild_artifacts` — rebuilds the catalog + `metrics.json` into S3
  (needs the `ARTIFACT_S3_BUCKET` repo variable).
- `deploy_frontend` — builds the SPA and publishes it to CloudFront
  (needs `WEB_BUCKET` + `CLOUDFRONT_DISTRIBUTION_ID` repo variables).

These require the deploy IAM role to also allow `s3:PutObject`/`ListBucket`/
`DeleteObject` on the artifact + web buckets and `cloudfront:CreateInvalidation`.

## Remote state (optional, recommended for teams)

State is local by default. To share it, uncomment the `backend "s3"` block in
`versions.tf`, create a state bucket + DynamoDB lock table once, then
`terraform init -migrate-state`.

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
