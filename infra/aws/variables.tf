variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short project name, used to prefix resource names."
  type        = string
  default     = "course-recommender"
}

variable "environment" {
  description = "Deployment environment (dev/staging/prod)."
  type        = string
  default     = "dev"
}

variable "container_image" {
  description = "Full ECR image URI (repo:tag) for the API. Set after first push."
  type        = string
  default     = ""
}

variable "container_port" {
  description = "Port the API container listens on."
  type        = number
  default     = 8000
}

variable "task_cpu" {
  description = "Fargate task CPU units (256, 512, 1024, ...)."
  type        = number
  default     = 1024
}

variable "task_memory" {
  description = "Fargate task memory (MiB)."
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "Number of ECS tasks to run."
  type        = number
  default     = 2
}

variable "llm_provider" {
  description = "LLM provider for the GenAI layer: 'bedrock' (native) or 'anthropic'."
  type        = string
  default     = "bedrock"
}

variable "llm_model" {
  description = "Model id. For Bedrock, e.g. anthropic.claude-3-5-sonnet-20240620-v1:0."
  type        = string
  default     = "anthropic.claude-3-5-sonnet-20240620-v1:0"
}

variable "enable_rerank" {
  description = "Enable cross-encoder reranking in the served API."
  type        = bool
  default     = false
}

variable "artifact_prefix" {
  description = "Key prefix for artifacts inside the S3 bucket."
  type        = string
  default     = "artifacts"
}
