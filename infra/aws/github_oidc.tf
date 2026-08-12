# GitHub Actions deploy role, assumed through OIDC.
#
# Lets the deploy-aws workflow build the image and roll the service without a
# long-lived access key in repository secrets: GitHub presents a short-lived
# token, AWS verifies it came from this repository, and hands back temporary
# credentials. It also moves the multi-gigabyte image push onto GitHub's
# network, which matters when the developer's uplink cannot sustain it.
#
# Set var.github_repository to "owner/repo" to create these; leave it empty to
# skip the whole block.

variable "github_repository" {
  description = "GitHub repo allowed to assume the deploy role, as owner/name. Empty disables OIDC."
  type        = string
  default     = ""
}

variable "github_branch" {
  description = "Branch allowed to assume the deploy role."
  type        = string
  default     = "main"
}

locals {
  oidc_enabled = var.github_repository != ""
}

# One OIDC provider per account. If the account already has one — a previous
# project, another team — import it rather than letting this fail:
#   terraform import aws_iam_openid_connect_provider.github \
#     arn:aws:iam::<account>:oidc-provider/token.actions.githubusercontent.com
resource "aws_iam_openid_connect_provider" "github" {
  count = local.oidc_enabled ? 1 : 0

  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

data "aws_iam_policy_document" "github_assume" {
  count = local.oidc_enabled ? 1 : 0

  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github[0].arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    # Scope to one branch of one repository. Without this, any GitHub repository
    # could assume the role.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:ref:refs/heads/${var.github_branch}"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  count = local.oidc_enabled ? 1 : 0

  name               = "${local.name_prefix}-github-deploy"
  description        = "Assumed by GitHub Actions to build, push and roll the service."
  assume_role_policy = data.aws_iam_policy_document.github_assume[0].json
}

data "aws_iam_policy_document" "github_deploy" {
  count = local.oidc_enabled ? 1 : 0

  # Push the API image. GetAuthorizationToken is account-wide by design: it
  # takes no resource, so it cannot be scoped to one repository.
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "EcrPush"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [aws_ecr_repository.api.arn]
  }

  # Roll the service onto the new image and wait for it to stabilize.
  statement {
    sid       = "EcsDeploy"
    actions   = ["ecs:UpdateService", "ecs:DescribeServices"]
    resources = [aws_ecs_service.api.id]
  }

  # Publish rebuilt artifacts and the compiled SPA.
  statement {
    sid     = "S3Publish"
    actions = ["s3:PutObject", "s3:DeleteObject", "s3:GetObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*",
      aws_s3_bucket.site.arn,
      "${aws_s3_bucket.site.arn}/*",
    ]
  }

  # Drop the CDN cache after publishing the SPA.
  statement {
    sid       = "CloudFrontInvalidate"
    actions   = ["cloudfront:CreateInvalidation"]
    resources = [aws_cloudfront_distribution.web.arn]
  }
}

resource "aws_iam_role_policy" "github_deploy" {
  count = local.oidc_enabled ? 1 : 0

  name   = "${local.name_prefix}-github-deploy"
  role   = aws_iam_role.github_deploy[0].id
  policy = data.aws_iam_policy_document.github_deploy[0].json
}

output "github_deploy_role_arn" {
  description = "Set as the AWS_DEPLOY_ROLE_ARN repository secret."
  value       = local.oidc_enabled ? aws_iam_role.github_deploy[0].arn : null
}
