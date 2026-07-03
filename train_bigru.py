#!/usr/bin/env python3
# train_bigru.py
# Loads collected gesture data, normalizes it, trains a BiGRU neural network,
# evaluates it, and saves the trained model.

import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn

# --- Constants ---
INTENTS = ["SELECT", "DISMISS", "SCROLL_UP", "SCROLL_DOWN"]
FRAMES_PER_SAMPLE = 30
LANDMARKS_PER_FRAME = 21
COORDS_PER_LANDMARK = 3
INPUT_SIZE = LANDMARKS_PER_FRAME * COORDS_PER_LANDMARK  # 63
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = len(INTENTS)
EPOCHS = 100
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.7
MODEL_SAVE_PATH = "models/bigru_intent.pth"

# --- Model Definition ---
class BiGRUClassifier(nn.Module):
    """Defines the BiGRU neural network architecture."""
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
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        output, _ = self.bigru(x)
        # Take the output from the last time step
        out = output[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# --- Data Processing Functions ---
def load_all_samples(data_dir="training_data"):
    """Load all JSON files from the data directory into one list."""
    json_pattern = os.path.join(data_dir, "*.json")
    file_paths = glob.glob(json_pattern)
    all_samples = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            samples_in_file = json.load(f)
            all_samples.extend(samples_in_file)

    print(f"Loaded {len(file_paths)} files with a total of {len(all_samples)} samples.")
    return all_samples

def normalize_sample(sample):
    """Make landmark coordinates relative to the wrist position."""
    for frame in sample["frames"]:
        wrist = frame["landmarks"][0]
        wrist_x, wrist_y, wrist_z = wrist['x'], wrist['y'], wrist['z']

        for landmark in frame["landmarks"]:
            landmark['x'] -= wrist_x
            landmark['y'] -= wrist_y
            landmark['z'] -= wrist_z
    return sample

def sample_to_tensor(sample):
    """Convert one sample dictionary into a PyTorch tensor of shape (30, 63)."""
    sequence = []
    for frame in sample["frames"]:
        frame_landmarks = []
        for landmark in frame["landmarks"]:
            frame_landmarks.extend([landmark['x'], landmark['y'], landmark['z']])
        sequence.append(frame_landmarks)

    # Ensure the sequence has exactly FRAMES_PER_SAMPLE frames
    if len(sequence) != FRAMES_PER_SAMPLE:
        # This case should ideally not happen with data_collector.py
        # but good to handle defensively.
        print(f"Warning: Sample has {len(sequence)} frames, expected {FRAMES_PER_SAMPLE}. Padding/truncating.")
        while len(sequence) < FRAMES_PER_SAMPLE:
            sequence.append([0.0] * INPUT_SIZE) # Pad with zeros
        sequence = sequence[:FRAMES_PER_SAMPLE] # Truncate

    return torch.FloatTensor(np.array(sequence, dtype=np.float32))

def prepare_data(samples):
    """Convert all samples into tensors and labels ready for training."""
    X = []
    y = []
    for sample in samples:
        normalized_sample = normalize_sample(sample)
        tensor = sample_to_tensor(normalized_sample)
        X.append(tensor)

        label_str = sample["label"]
        y.append(INTENTS.index(label_str))

    X_tensor = torch.stack(X)
    y_tensor = torch.LongTensor(y)

    print(f"Prepared data tensors: X shape {X_tensor.shape}, y shape {y_tensor.shape}")
    return X_tensor, y_tensor

def split_data(X, y, train_split=TRAIN_SPLIT):
    """Split data into training and test sets."""
    num_samples = len(X)
    split_index = int(num_samples * train_split)

    shuffled_indices = torch.randperm(num_samples)
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    X_train = X_shuffled[:split_index]
    y_train = y_shuffled[:split_index]
    X_test = X_shuffled[split_index:]
    y_test = y_shuffled[split_index:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# --- Training & Evaluation Functions ---
def train_model(model, X_train, y_train):
    """Run the training loop for EPOCHS iterations."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {loss.item():.4f}")

    print("--- Training complete. ---\n")

def evaluate_model(model, X_test, y_test):
    """Measure model accuracy on the test set and print per-intent results."""
    print("--- Evaluating Model ---")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == y_test).sum().item()
        total = y_test.size(0)
        accuracy = (correct / total) * 100
        print(f"Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")

        print("\nPer-intent accuracy:")
        for i, intent in enumerate(INTENTS):
            intent_mask = (y_test == i)
            total_intent = intent_mask.sum().item()
            if total_intent == 0:
                print(f"{intent}: 0/0 (N/A)")
                continue
            
            correct_intent = (predictions[intent_mask] == y_test[intent_mask]).sum().item()
            percent_intent = (correct_intent / total_intent) * 100
            print(f"{intent}: {correct_intent}/{total_intent} ({percent_intent:.1f}%)")
    print("--- Evaluation complete. ---\n")

def save_model(model):
    """Save trained model weights to disk."""
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# --- Main Execution ---
def main():
    """Main function to run the full training pipeline."""
    print("--- HandIntent BiGRU Trainer ---")
    samples = load_all_samples()
    if not samples:
        print("No samples found. Exiting.")
        return

    X, y = prepare_data(samples)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = BiGRUClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    print("\nModel Architecture:")
    print(model)

    train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)

if __name__ == "__main__":
    main()