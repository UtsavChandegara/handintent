#!/usr/bin/env python3
# train_bigru_cloud.py
# Cloud version of BiGRU trainer.
# Downloads data from S3, trains, uploads model back to S3.
# Runs on EC2 with HandIntentEC2Role IAM permissions.

import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import boto3
import datetime
import subprocess

# Constants
BUCKET_NAME = "handintent-data"
REGION = "ap-south-1"
S3_TRAINING_PREFIX = "training_data/"
S3_MODELS_PREFIX = "models/"
LOCAL_TRAINING_DIR = "/tmp/training_data"
LOCAL_MODEL_PATH = "/tmp/bigru_intent.pth"

INTENTS = ["SELECT", "DISMISS", "SCROLL_UP", "SCROLL_DOWN"]
FRAMES_PER_SAMPLE = 30
LANDMARKS_PER_FRAME = 21
COORDS_PER_LANDMARK = 3
INPUT_SIZE = LANDMARKS_PER_FRAME * COORDS_PER_LANDMARK
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = len(INTENTS)
EPOCHS = 100
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.7


class BiGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiGRUClassifier, self).__init__()
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        output, _ = self.bigru(x)
        out = output[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def download_training_data(s3_client):
    print("Downloading training data from S3...")
    os.makedirs(LOCAL_TRAINING_DIR, exist_ok=True)

    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=S3_TRAINING_PREFIX
    )

    if "Contents" not in response:
        print("No training data found in S3.")
        return False

    downloaded = 0
    for obj in response["Contents"]:
        key = obj["Key"]
        if not key.endswith(".json"):
            continue
        filename = os.path.basename(key)
        local_path = os.path.join(LOCAL_TRAINING_DIR, filename)
        s3_client.download_file(BUCKET_NAME, key, local_path)
        print(f"  ✓ Downloaded {filename}")
        downloaded += 1

    print(f"Downloaded {downloaded} training files.")
    return downloaded > 0


def load_all_samples():
    all_samples = []
    files = glob.glob(f"{LOCAL_TRAINING_DIR}/*.json")
    for f in files:
        with open(f, "r") as fp:
            all_samples.extend(json.load(fp))
    print(f"Loaded {len(all_samples)} total samples.")
    return all_samples


def normalize_sample(sample):
    for frame in sample["frames"]:
        wrist = frame["landmarks"][0]
        wx, wy, wz = wrist["x"], wrist["y"], wrist["z"]
        for lm in frame["landmarks"]:
            lm["x"] -= wx
            lm["y"] -= wy
            lm["z"] -= wz
    return sample


def sample_to_tensor(sample):
    sequence = []
    for frame in sample["frames"]:
        flat = []
        for lm in frame["landmarks"]:
            flat.extend([lm["x"], lm["y"], lm["z"]])
        sequence.append(flat)
    while len(sequence) < FRAMES_PER_SAMPLE:
        sequence.append([0.0] * INPUT_SIZE)
    sequence = sequence[:FRAMES_PER_SAMPLE]
    return torch.FloatTensor(np.array(sequence, dtype=np.float32))


def prepare_data(samples):
    X, y = [], []
    for sample in samples:
        normalize_sample(sample)
        X.append(sample_to_tensor(sample))
        y.append(INTENTS.index(sample["label"]))
    X = torch.stack(X)
    y = torch.LongTensor(y)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    return X, y


def split_data(X, y):
    n = len(X)
    split = int(n * TRAIN_SPLIT)
    idx = torch.randperm(n)
    X, y = X[idx], y[idx]
    return X[:split], X[split:], y[:split], y[split:]


def train_model(model, X_train, y_train):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    print("\n--- Training ---")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} Loss: {loss.item():.4f}")
    print("Training complete.")


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)
        correct = (preds == y_test).sum().item()
        accuracy = correct / len(y_test) * 100
        print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{len(y_test)})")
        for i, intent in enumerate(INTENTS):
            mask = y_test == i
            total = mask.sum().item()
            if total == 0:
                continue
            correct_i = (preds[mask] == y_test[mask]).sum().item()
            print(f"  {intent}: {correct_i}/{total} ({correct_i/total*100:.1f}%)")
    return accuracy


def upload_model(s3_client, accuracy):
    print("\nUploading model to S3...")
    timestamp = int(datetime.datetime.now().timestamp())
    versioned = f"bigru_intent_{timestamp}_acc{int(accuracy)}.pth"
    s3_key = S3_MODELS_PREFIX + versioned
    s3_client.upload_file(LOCAL_MODEL_PATH, BUCKET_NAME, s3_key)
    print(f"  ✓ Model uploaded as {versioned}")

    # Also upload as latest
    s3_client.upload_file(
        LOCAL_MODEL_PATH,
        BUCKET_NAME,
        S3_MODELS_PREFIX + "bigru_intent_latest.pth"
    )
    print(f"  ✓ Also saved as bigru_intent_latest.pth")


def main():
    print("=== HandIntent Cloud Trainer ===")
    print(f"Started at: {datetime.datetime.now()}")

    s3_client = boto3.client("s3", region_name=REGION)

    if not download_training_data(s3_client):
        print("No data to train on. Exiting.")
        return

    samples = load_all_samples()
    if not samples:
        print("No samples loaded. Exiting.")
        return

    X, y = prepare_data(samples)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {len(X_train)} Test: {len(X_test)}")

    model = BiGRUClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    train_model(model, X_train, y_train)
    torch.save(model.state_dict(), LOCAL_MODEL_PATH)

    accuracy = evaluate_model(model, X_test, y_test)
    upload_model(s3_client, accuracy)

    print(f"\nFinished at: {datetime.datetime.now()}")
    print("Training pipeline complete.")


if __name__ == "__main__":
    main()
