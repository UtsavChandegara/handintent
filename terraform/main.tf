# main.tf
# HandIntent Infrastructure as Code
# Provisions all AWS resources for the HandIntent project

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "ap-south-1"
}

# ─── S3 Bucket ───────────────────────────────────────────────
resource "aws_s3_bucket" "handintent_data" {
  bucket = "handintent-data"

  tags = {
    Name        = "HandIntent Data"
    Project     = "HandIntent"
    Environment = "production"
  }
}

resource "aws_s3_object" "training_data_prefix" {
  bucket = aws_s3_bucket.handintent_data.id
  key    = "training_data/"
}

resource "aws_s3_object" "models_prefix" {
  bucket = aws_s3_bucket.handintent_data.id
  key    = "models/"
}

resource "aws_s3_object" "logs_prefix" {
  bucket = aws_s3_bucket.handintent_data.id
  key    = "logs/"
}

# ─── IAM Role for EC2 ────────────────────────────────────────
resource "aws_iam_role" "ec2_role" {
  name = "HandIntentEC2Role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_s3_access" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "HandIntentEC2Profile"
  role = aws_iam_role.ec2_role.name
}

# ─── IAM Role for Lambda ──────────────────────────────────────
resource "aws_iam_role" "lambda_role" {
  name = "HandIntentLambdaRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_ec2_access" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2FullAccess"
}

resource "aws_iam_role_policy_attachment" "lambda_logs_access" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

# ─── ECR Repository ──────────────────────────────────────────
resource "aws_ecr_repository" "handintent_training" {
  name = "handintent-training"

  tags = {
    Project = "HandIntent"
  }
}

# ─── Outputs ─────────────────────────────────────────────────
output "s3_bucket_name" {
  value = aws_s3_bucket.handintent_data.bucket
}

output "ecr_repository_uri" {
  value = aws_ecr_repository.handintent_training.repository_url
}

output "ec2_role_arn" {
  value = aws_iam_role.ec2_role.arn
}

output "lambda_role_arn" {
  value = aws_iam_role.lambda_role.arn
}
