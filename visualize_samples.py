#!/usr/bin/env python3
# visualize_samples.py
# Standalone analysis script for visualizing collected gesture data.

import json
import os
import glob
import matplotlib.pyplot as plt

def load_all_samples(data_dir="training_data"):
    """Load all JSON files from the data directory and combine them into one list."""
    json_pattern = os.path.join(data_dir, "*.json")
    file_paths = glob.glob(json_pattern)
    all_samples = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            samples_in_file = json.load(f)
            all_samples.extend(samples_in_file)
    
    print(f"Loaded {len(file_paths)} files with a total of {len(all_samples)} samples.")
    return all_samples

def filter_by_label(samples, label):
    """Filter a list of samples to only include those with a matching label."""
    return [sample for sample in samples if sample["label"] == label]

def extract_landmark_sequence(sample, landmark_index):
    """Extract x, y, z coordinates of a specific landmark across all frames in a sample."""
    x_coords = []
    y_coords = []
    z_coords = []

    for frame in sample["frames"]:
        landmark = frame["landmarks"][landmark_index]
        x_coords.append(landmark["x"])
        y_coords.append(landmark["y"])
        z_coords.append(landmark["z"])
        
    return x_coords, y_coords, z_coords

def plot_select_samples(samples):
    """Create and display two matplotlib figures for SELECT samples."""
    
    # --- Figure 1: Index Fingertip Movement (Landmark 8) ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig1.suptitle("SELECT Gesture: Index Fingertip (Landmark 8) Movement")

    # Left subplot: X coordinate
    ax1.set_title("Index Fingertip X over 30 frames")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("X coordinate (normalized)")

    # Right subplot: Y coordinate
    ax2.set_title("Index Fingertip Y over 30 frames")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Y coordinate (normalized)")

    for i, sample in enumerate(samples):
        x_coords, y_coords, _ = extract_landmark_sequence(sample, 8)
        ax1.plot(range(len(x_coords)), x_coords, label=f"Sample {i+1}")
        ax2.plot(range(len(y_coords)), y_coords, label=f"Sample {i+1}")

    ax1.legend()
    ax2.legend()
    plt.savefig("select_fingertip.png")
    plt.close()

    # --- Figure 2: Wrist Z Movement (Landmark 0) ---
    plt.figure(figsize=(10, 6))
    plt.title("SELECT Gesture: Wrist Z (Depth) over 30 frames")
    plt.xlabel("Frame")
    plt.ylabel("Z coordinate (depth)")

    for i, sample in enumerate(samples):
        _, _, z_coords = extract_landmark_sequence(sample, 0)
        plt.plot(range(len(z_coords)), z_coords, label=f"Sample {i+1}")
    
    plt.legend()
    plt.savefig("select_wrist_z.png")
    plt.close()

def main():
    """Load, filter, and plot gesture data."""
    all_samples = load_all_samples()
    select_samples = filter_by_label(all_samples, "SELECT")

    if not select_samples:
        print("No SELECT samples found in training_data/")
        return

    print(f"Found {len(select_samples)} SELECT samples. Plotting...")
    plot_select_samples(select_samples)

if __name__ == "__main__":
    main()