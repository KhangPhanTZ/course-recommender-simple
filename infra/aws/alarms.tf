# ---------------------------------------------------------------------------
# CloudWatch alarms for basic production observability. Set var.alarm_email to
# receive notifications (an SNS topic + email subscription are created and used
# as the alarm action); leave it empty to just surface alarm state in the console.
# ---------------------------------------------------------------------------
locals {
  alarm_enabled = var.alarm_email != ""
  alarm_actions = local.alarm_enabled ? [aws_sns_topic.alerts[0].arn] : []
}

resource "aws_sns_topic" "alerts" {
  count = local.alarm_enabled ? 1 : 0
  name  = "${local.name_prefix}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  count     = local.alarm_enabled ? 1 : 0
  topic_arn = aws_sns_topic.alerts[0].arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

# Sustained high CPU (autoscaling should react first; this flags saturation).
resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "${local.name_prefix}-cpu-high"
  namespace           = "AWS/ECS"
  metric_name         = "CPUUtilization"
  statistic           = "Average"
  period              = 60
  evaluation_periods  = 5
  threshold           = 85
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    ClusterName = aws_ecs_cluster.this.name
    ServiceName = aws_ecs_service.api.name
  }
  alarm_actions = local.alarm_actions
  ok_actions    = local.alarm_actions
}

# Target 5xx errors coming from the app.
resource "aws_cloudwatch_metric_alarm" "target_5xx" {
  alarm_name          = "${local.name_prefix}-target-5xx"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  statistic           = "Sum"
  period              = 60
  evaluation_periods  = 3
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    LoadBalancer = aws_lb.this.arn_suffix
    TargetGroup  = aws_lb_target_group.api.arn_suffix
  }
  alarm_actions = local.alarm_actions
  ok_actions    = local.alarm_actions
}

# No healthy tasks behind the load balancer.
resource "aws_cloudwatch_metric_alarm" "unhealthy_hosts" {
  alarm_name          = "${local.name_prefix}-unhealthy-hosts"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "UnHealthyHostCount"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 3
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    LoadBalancer = aws_lb.this.arn_suffix
    TargetGroup  = aws_lb_target_group.api.arn_suffix
  }
  alarm_actions = local.alarm_actions
  ok_actions    = local.alarm_actions
}
