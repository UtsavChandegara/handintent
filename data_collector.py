# data_collector.py
# Standalone script for collecting labeled hand gesture data.
# Author: Utsav

import cv2
import time
import os
import json
import mediapipe.python.solutions.hands as mp_hands_module

# Constants
FRAMES_PER_SAMPLE = 30
INTENTS = ["SELECT", "DISMISS", "SCROLL_UP", "SCROLL_DOWN"]
DATA_DIR = "training_data"


def initialize_collector():
    """Set up MediaPipe, webcam, and data directory."""
    hands = mp_hands_module.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)

    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = f"{DATA_DIR}/gestures_{int(time.time())}.json"
    all_samples = []

    print("Data collector ready.")
    return hands, cap, output_path, all_samples


def extract_landmarks(hand_landmarks, handedness_label):
    """Convert MediaPipe landmark object into a clean dictionary for JSON storage."""
    landmarks_list = []
    for lm in hand_landmarks.landmark:
        landmarks_list.append({
            "x": round(lm.x, 4),
            "y": round(lm.y, 4),
            "z": round(lm.z, 4)
        })

    return {
        "landmarks": landmarks_list,
        "handedness": handedness_label
    }


def record_gesture_sample(cap, hands, label, context):
    """Record exactly FRAMES_PER_SAMPLE frames of one gesture and return as a labeled sample."""
    # Step 1: Countdown
    for i in range(3, 0, -1):
        print(f"Recording {label} in {i}...")
        time.sleep(1)
    print("GO!")

    # Step 2: Recording message
    print("Recording... perform gesture now")

    # Step 3: Initialize frames list
    frames = []

    # Step 4: Recording loop
    while len(frames) < FRAMES_PER_SAMPLE:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            handedness_label = result.multi_handedness[0].classification[0].label

            frame_data = extract_landmarks(hand_landmarks, handedness_label)
            frame_data["frame_number"] = len(frames)
            frames.append(frame_data)

        # Draw progress on screen
        progress_text = f"Recording: {len(frames)}/{FRAMES_PER_SAMPLE}"
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording interrupted by user.")
            return None # Indicate interruption

    # Step 5: Confirmation
    print(f"Sample recorded. {len(frames)} frames captured.")

    # Step 6: Return sample
    return {
        "label": label,
        "context": context,
        "frames": frames
    }


def save_samples(all_samples, output_path):
    """Save all collected samples to JSON file."""
    if not all_samples:
        print("No samples to save.")
        return

    with open(output_path, 'w') as f:
        json.dump(all_samples, f, indent=2)
    print(f"Saved {len(all_samples)} samples to {output_path}")


def main():
    """Interactive collection loop. User selects which gesture to record repeatedly until done."""
    # Step 1: Initialize
    hands, cap, output_path, all_samples = initialize_collector()

    # Step 2: Print menu
    print("\nHandIntent Data Collector")
    print("=========================")
    print("Intents available:")
    for i, intent in enumerate(INTENTS):
        print(f"{i}: {intent}")
    print("Q: Quit and save")

    # Step 3: Main loop
    while True:
        choice = input("\nEnter intent number to record (or Q to quit): ").strip().lower()

        if choice == 'q':
            break

        if choice.isdigit() and 0 <= int(choice) < len(INTENTS):
            label = INTENTS[int(choice)]
            sample = record_gesture_sample(cap, hands, label, "READING")
            if sample:
                all_samples.append(sample)
                print(f"Total samples collected: {len(all_samples)}")
        else:
            print("Invalid input. Try again.")

    # Step 4: Cleanup and save
    save_samples(all_samples, output_path)
    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete.")


if __name__ == "__main__":
    main()