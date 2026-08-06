# ---------------------------------------------------------------------------
# CloudFront CDN: serves the React SPA over HTTPS and proxies /api/* to the ALB,
# so the whole app is same-origin and needs no custom domain or ACM cert.
#
#   viewer ──https──> CloudFront ──┬── default        ──> S3 (static SPA)
#                                  └── /api/*          ──> ALB (FastAPI, HTTP)
#
# A CloudFront Function strips the /api prefix before the request reaches the
# ALB, so the API keeps serving at /health, /recommend, … unchanged.
# ---------------------------------------------------------------------------

# --- Private bucket holding the built SPA ----------------------------------
resource "aws_s3_bucket" "site" {
  bucket_prefix = "${local.name_prefix}-web-"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "site" {
  bucket                  = aws_s3_bucket.site.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "site" {
  bucket = aws_s3_bucket.site.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# --- CloudFront reads S3 privately via Origin Access Control ----------------
resource "aws_cloudfront_origin_access_control" "site" {
  name                              = "${local.name_prefix}-oac"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# --- Strip the /api prefix before forwarding to the ALB origin --------------
resource "aws_cloudfront_function" "strip_api" {
  name    = "${local.name_prefix}-strip-api"
  runtime = "cloudfront-js-2.0"
  comment = "Rewrite /api/* -> /* for the ALB origin"
  publish = true
  code    = <<-EOT
    function handler(event) {
      var request = event.request;
      request.uri = request.uri.replace(/^\/api/, "");
      if (request.uri === "") { request.uri = "/"; }
      return request;
    }
  EOT
}

# Managed policies (by name) so we don't hardcode UUIDs.
data "aws_cloudfront_cache_policy" "optimized" {
  name = "Managed-CachingOptimized"
}

data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_origin_request_policy" "all_viewer" {
  name = "Managed-AllViewerExceptHostHeader"
}

resource "aws_cloudfront_distribution" "web" {
  enabled             = true
  default_root_object = "index.html"
  comment             = "${local.name_prefix} web + API"
  price_class         = "PriceClass_100" # NA + EU edges (cheapest)

  # Static SPA origin.
  origin {
    origin_id                = "s3-site"
    domain_name              = aws_s3_bucket.site.bucket_regional_domain_name
    origin_access_control_id = aws_cloudfront_origin_access_control.site.id
  }

  # API origin (the public ALB, HTTP).
  origin {
    origin_id   = "alb-api"
    domain_name = aws_lb.this.dns_name
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  # Default: serve the SPA from S3.
  default_cache_behavior {
    target_origin_id       = "s3-site"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true
    cache_policy_id        = data.aws_cloudfront_cache_policy.optimized.id
  }

  # /api/*: proxy to the ALB, no caching, forward everything.
  ordered_cache_behavior {
    path_pattern             = "/api/*"
    target_origin_id         = "alb-api"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    compress                 = true
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id

    function_association {
      event_type   = "viewer-request"
      function_arn = aws_cloudfront_function.strip_api.arn
    }
  }

  # SPA client-side routing: a missing object (403 from OAC) serves index.html.
  # API 404s come back as 404 from the ALB and are not rewritten.
  custom_error_response {
    error_code            = 403
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 10
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true # HTTPS via *.cloudfront.net
  }
}

# --- Let CloudFront (this distribution only) read the site bucket -----------
data "aws_iam_policy_document" "site" {
  statement {
    sid       = "AllowCloudFrontRead"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.site.arn}/*"]
    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = [aws_cloudfront_distribution.web.arn]
    }
  }
}

resource "aws_s3_bucket_policy" "site" {
  bucket = aws_s3_bucket.site.id
  policy = data.aws_iam_policy_document.site.json
}
