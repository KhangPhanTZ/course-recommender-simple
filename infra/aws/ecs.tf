resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name_prefix}-api"
  retention_in_days = 14
}

resource "aws_ecs_cluster" "this" {
  name = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

locals {
  # Fall back to a public placeholder until the first image is pushed, so
  # `terraform plan` works before CI has built anything.
  image = var.container_image != "" ? var.container_image : "${aws_ecr_repository.api.repository_url}:latest"
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name_prefix}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name      = "api"
    image     = local.image
    essential = true
    portMappings = [{
      containerPort = var.container_port
      protocol      = "tcp"
    }]
    environment = [
      { name = "ARTIFACT_STORE", value = "s3" },
      { name = "ARTIFACT_S3_BUCKET", value = aws_s3_bucket.artifacts.bucket },
      { name = "ARTIFACT_S3_PREFIX", value = var.artifact_prefix },
      { name = "AWS_REGION", value = var.aws_region },
      { name = "LLM_PROVIDER", value = var.llm_provider },
      { name = "LLM_MODEL", value = var.llm_model },
      { name = "LLM_ENABLED", value = "true" },
      { name = "ENABLE_RERANK", value = tostring(var.enable_rerank) },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.api.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "api"
      }
    }
  }])
}

resource "aws_ecs_service" "api" {
  name            = "${local.name_prefix}-api"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = true # default subnets are public; needed to pull images
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = var.container_port
  }

  # CI updates the service with a new image tag; ignore drift on the task def.
  # Autoscaling manages desired_count at runtime, so ignore it here too.
  lifecycle {
    ignore_changes = [task_definition, desired_count]
  }

  depends_on = [aws_lb_listener.http]
}
