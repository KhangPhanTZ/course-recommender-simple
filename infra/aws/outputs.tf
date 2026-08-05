output "api_url" {
  description = "Public base URL of the recommender API."
  value       = "http://${aws_lb.this.dns_name}"
}

output "ecr_repository_url" {
  description = "Push container images here."
  value       = aws_ecr_repository.api.repository_url
}

output "artifacts_bucket" {
  description = "S3 bucket for model artifacts (set ARTIFACT_S3_BUCKET to this)."
  value       = aws_s3_bucket.artifacts.bucket
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "ecs_service_name" {
  value = aws_ecs_service.api.name
}

output "aws_region" {
  description = "Region the stack is deployed in."
  value       = var.aws_region
}
