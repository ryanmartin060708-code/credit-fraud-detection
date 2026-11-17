terraform {
  required_version = "~> 1.10.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.90.0" }
  }
}
provider "aws" {
  region                      = var.region
  access_key                  = var.offline_plan ? "offline-placeholder" : null
  secret_key                  = var.offline_plan ? "offline-placeholder" : null
  skip_credentials_validation = var.offline_plan
  skip_requesting_account_id  = var.offline_plan
  skip_metadata_api_check     = var.offline_plan
  skip_region_validation      = var.offline_plan
  default_tags { tags = { Project = "credit-fraud-detection" } }
}
variable "offline_plan" {
  type        = bool
  default     = false
  description = "Schema/graph planning only, with dummy credentials and no AWS validation. Never apply in this mode."
}
variable "region" {
  type    = string
  default = "ap-south-1"
}
variable "image" {
  type        = string
  description = "Reachable container image URI; publish the tested image before any deployment."
}
variable "operator_cidr" {
  type        = string
  description = "Trusted operator or private gateway IPv4 /32; never exposes the demo to everyone."
  validation {
    condition     = can(cidrhost(var.operator_cidr, 0)) && endswith(var.operator_cidr, "/32")
    error_message = "Use a trusted IPv4 /32 CIDR."
  }
}
resource "aws_vpc" "app" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_hostnames = true
}
resource "aws_subnet" "app" {
  vpc_id     = aws_vpc.app.id
  cidr_block = "10.42.1.0/24"
}
resource "aws_internet_gateway" "app" { vpc_id = aws_vpc.app.id }
resource "aws_route_table" "app" {
  vpc_id = aws_vpc.app.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.app.id
  }
}
resource "aws_route_table_association" "app" {
  subnet_id      = aws_subnet.app.id
  route_table_id = aws_route_table.app.id
}
resource "aws_security_group" "app" {
  name_prefix = "credit-fraud-detection-"
  vpc_id      = aws_vpc.app.id
  dynamic "ingress" {
    for_each = [8501, 9000]
    content {
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "tcp"
      cidr_blocks = [var.operator_cidr]
    }
  }
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/credit-fraud-detection"
  retention_in_days = 7
}
locals {
  assume_role = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}
resource "aws_iam_role" "execution" {
  name_prefix        = "credit-fraud-detection-execution-"
  assume_role_policy = local.assume_role
}
resource "aws_iam_role_policy" "logs" {
  role   = aws_iam_role.execution.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "${aws_cloudwatch_log_group.app.arn}:*" }] })
}
resource "aws_iam_role" "task" {
  name_prefix        = "credit-fraud-detection-task-"
  assume_role_policy = local.assume_role
}
resource "aws_ecs_cluster" "app" { name = "credit-fraud-detection" }
resource "aws_ecs_task_definition" "app" {
  family                   = "credit-fraud-detection"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  container_definitions = jsonencode([{
    name         = "app", image = var.image, essential = true,
    portMappings = [for port in [8501, 9000] : { containerPort = port, protocol = "tcp" }],
    environment  = [{ name = "AWS_REGION", value = var.region }],
    healthCheck = {
      command  = ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9000/health', timeout=3)"],
      interval = 30, timeout = 5, retries = 3, startPeriod = 120
    },
    logConfiguration = { logDriver = "awslogs", options = {
      awslogs-group         = aws_cloudwatch_log_group.app.name,
      awslogs-region        = var.region,
      awslogs-stream-prefix = "app"
    } }
  }])
}
resource "aws_ecs_service" "app" {
  name            = "credit-fraud-detection"
  cluster         = aws_ecs_cluster.app.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = [aws_subnet.app.id]
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true
  }
  depends_on = [aws_iam_role_policy.logs, aws_route_table_association.app]
}
output "cluster_name" { value = aws_ecs_cluster.app.name }
