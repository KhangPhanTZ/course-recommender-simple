# AWS Infrastructure (Terraform)

Provisions a production-shaped deployment of the course recommender on AWS:

```
Internet ──▶ ALB (HTTP :80) ──▶ ECS Fargate service (FastAPI :8000, N tasks)
                                   │  task IAM role
                                   ├──▶ S3 (model artifacts, read-only)
                                   └──▶ Amazon Bedrock (Claude, GenAI/RAG)
ECR  ◀── container images        CloudWatch Logs + Container Insights
```

## Resources created

| File | Resources |
|------|-----------|
| `network.tf` | Default VPC/subnet lookups, ALB + service security groups |
| `ecr.tf`     | ECR repository + lifecycle policy |
| `s3.tf`      | Private, encrypted, versioned artifacts bucket |
| `iam.tf`     | ECS execution role + task role (S3 read, Bedrock invoke) |
| `alb.tf`     | Application Load Balancer, target group, listener |
| `ecs.tf`     | ECS cluster, Fargate task definition, service, log group |

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

echo "API: $(cd infra/aws && terraform output -raw api_url)"
```

## Cost & teardown

Fargate tasks + ALB accrue hourly cost. Tear everything down with:

```bash
cd infra/aws && terraform destroy
```

## Notes

- Uses the **default VPC** for simplicity. For production, front the ALB with
  HTTPS (ACM certificate + `:443` listener) and deploy into private subnets
  with a NAT gateway.
- `bedrock` is the default GenAI provider — no API key needed, the task IAM
  role authorizes `bedrock:InvokeModel`. Switch `llm_provider` to `anthropic`
  and inject `ANTHROPIC_API_KEY` (via SSM/Secrets Manager) to use the Claude API.
