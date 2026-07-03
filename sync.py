#!/usr/bin/env python3
# sync.py
# Manages synchronization of data, logs, and models with AWS S3.

import boto3
import os
import glob
import datetime
from pathlib import Path

# --- Constants ---
BUCKET_NAME = "handintent-data"
REGION = "ap-south-1"
S3_TRAINING_PREFIX = "training_data/"
S3_MODELS_PREFIX = "models/"
S3_LOGS_PREFIX = "logs/"
LOCAL_TRAINING_DIR = "training_data"
LOCAL_MODELS_DIR = "models"
LOCAL_LOGS_DIR = "logs"
LOCAL_MODEL_PATH = "models/bigru_intent.pth"

def get_s3_client():
    """Create and return a boto3 S3 client configured for the correct region."""
    return boto3.client('s3', region_name=REGION)

def get_existing_s3_keys(s3_client, prefix):
    """Get list of all file keys already in S3 under a given prefix."""
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)
        keys = []
        for page in pages:
            if "Contents" in page:
                for obj in page['Contents']:
                    keys.append(obj['Key'])
        return keys
    except Exception as e:
        print(f"  - Error listing S3 objects for prefix '{prefix}': {e}")
        return []

def upload_training_data(s3_client):
    """Upload all local JSON files from training_data/ to S3. Skip files already there."""
    print("Uploading training data...")
    s3_keys = get_existing_s3_keys(s3_client, S3_TRAINING_PREFIX)
    local_files = glob.glob(f"{LOCAL_TRAINING_DIR}/*.json")
    
    if not local_files:
        print("  - No training data found locally.")
        return

    uploaded_count = 0
    for local_path in local_files:
        filename = os.path.basename(local_path)
        s3_key = f"{S3_TRAINING_PREFIX}{filename}"
        
        if s3_key in s3_keys:
            print(f"  - {filename} already exists, skipping")
        else:
            try:
                s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
                print(f"  ✓ {filename} → S3")
                uploaded_count += 1
            except Exception as e:
                print(f"  ✗ Failed to upload {filename}: {e}")

    print(f"Training data upload complete. {uploaded_count}/{len(local_files)} new files uploaded.")

def upload_session_logs(s3_client):
    """Upload all local CSV files from logs/ to S3. Skip files already there."""
    print("Uploading session logs...")
    s3_keys = get_existing_s3_keys(s3_client, S3_LOGS_PREFIX)
    local_files = glob.glob(f"{LOCAL_LOGS_DIR}/*.csv")

    if not local_files:
        print("  - No session logs found locally.")
        return

    uploaded_count = 0
    for local_path in local_files:
        filename = os.path.basename(local_path)
        s3_key = f"{S3_LOGS_PREFIX}{filename}"

        if s3_key in s3_keys:
            print(f"  - {filename} already exists, skipping")
        else:
            try:
                s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
                print(f"  ✓ {filename} → S3")
                uploaded_count += 1
            except Exception as e:
                print(f"  ✗ Failed to upload {filename}: {e}")
    
    print(f"Session log upload complete. {uploaded_count}/{len(local_files)} new files uploaded.")

def get_latest_s3_model(s3_client):
    """Find the most recently uploaded model file in S3."""
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_MODELS_PREFIX)
        if "Contents" not in response:
            return None

        model_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.pth')]
        if not model_files:
            return None

        latest_model = sorted(model_files, key=lambda x: x['LastModified'], reverse=True)[0]
        return latest_model
    except Exception as e:
        print(f"  - Error finding latest model on S3: {e}")
        return None

def download_latest_model(s3_client):
    """Check if S3 has a newer model than local. Download if yes."""
    print("Checking for model updates...")
    latest_s3_model = get_latest_s3_model(s3_client)

    if not latest_s3_model:
        print("  - No model found in S3.")
        return

    s3_key = latest_s3_model['Key']
    s3_last_modified = latest_s3_model['LastModified']
    local_model_file = Path(LOCAL_MODEL_PATH)

    if not local_model_file.exists():
        print("  - No local model found. Downloading from S3...")
        try:
            os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
            s3_client.download_file(BUCKET_NAME, s3_key, str(local_model_file))
            print(f"  ✓ Downloaded {os.path.basename(s3_key)} successfully.")
        except Exception as e:
            print(f"  ✗ Failed to download model: {e}")
        return

    local_mtime_unix = os.path.getmtime(local_model_file)
    local_last_modified = datetime.datetime.fromtimestamp(local_mtime_unix, tz=datetime.timezone.utc)

    if s3_last_modified > local_last_modified:
        print("  - Newer model found on S3. Downloading...")
        try:
            s3_client.download_file(BUCKET_NAME, s3_key, str(local_model_file))
            print(f"  ✓ Downloaded {os.path.basename(s3_key)} successfully.")
        except Exception as e:
            print(f"  ✗ Failed to download model: {e}")
    else:
        print("  - Local model is up to date. No download needed.")

def upload_model(s3_client):
    """Upload current local trained model to S3 with timestamp in filename for versioning."""
    print("Uploading local model...")
    local_model_file = Path(LOCAL_MODEL_PATH)
    if not local_model_file.exists():
        print("  - No local model found to upload.")
        return

    try:
        timestamp = int(datetime.datetime.now().timestamp())
        versioned_filename = f"bigru_intent_{timestamp}.pth"
        s3_key = f"{S3_MODELS_PREFIX}{versioned_filename}"
        
        s3_client.upload_file(str(local_model_file), BUCKET_NAME, s3_key)
        print(f"  ✓ Model uploaded to S3 as {versioned_filename}")
    except Exception as e:
        print(f"  ✗ Failed to upload model: {e}")

def main():
    """Run full sync — upload data, upload logs, check and download model."""
    print("HandIntent S3 Sync")
    print("==================")
    
    try:
        s3_client = get_s3_client()
    except Exception as e:
        print(f"Failed to connect to AWS. Check your credentials. Error: {e}")
        return

    upload_training_data(s3_client)
    print()
    upload_session_logs(s3_client)
    print()
    download_latest_model(s3_client)
    print()

    try:
        choice = input("Upload current local model to S3? (y/n): ").strip().lower()
        if choice == 'y':
            upload_model(s3_client)
        else:
            print("Model upload skipped.")
    except (EOFError, KeyboardInterrupt):
        print("\nModel upload skipped.")

    print("\nSync complete.")

if __name__ == "__main__":
    main()