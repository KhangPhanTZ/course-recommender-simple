terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment to store state remotely (recommended for teams). Create the
  # bucket + lock table once, then run `terraform init -migrate-state`.
  # backend "s3" {
  #   bucket         = "my-tf-state-bucket"
  #   key            = "course-recommender/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "my-tf-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
