# lambda_trigger.py
# Triggered by S3 when new training data arrives.
# Starts EC2 training instance automatically.

import boto3
import os

INSTANCE_ID = "i-0776a9cbca9fb44c8"
REGION = "ap-south-1"

def lambda_handler(event, context):
    print(f"S3 event received: {event}")
    
    # Check if the uploaded file is a JSON training file
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        
        print(f"New file detected: s3://{bucket}/{key}")
        
        # Only trigger training for JSON files in training_data/
        if not key.startswith("training_data/") or not key.endswith(".json"):
            print("Not a training data file. Skipping.")
            continue
        
        # Start EC2 instance
        ec2 = boto3.client("ec2", region_name=REGION)
        
        # Check current instance state
        response = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
        state = response["Reservations"][0]["Instances"][0]["State"]["Name"]
        print(f"EC2 instance current state: {state}")
        
        if state == "stopped":
            ec2.start_instances(InstanceIds=[INSTANCE_ID])
            print(f"EC2 instance {INSTANCE_ID} started successfully.")
        elif state == "running":
            print("EC2 instance already running. Training may already be in progress.")
        else:
            print(f"EC2 instance in state: {state}. Cannot start.")
    
    return {
        "statusCode": 200,
        "body": "Lambda trigger executed successfully."
    }
