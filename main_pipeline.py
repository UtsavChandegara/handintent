# main_pipeline.py
# Integrates all stages of the HandIntent MVP pipeline.
# Author: Utsav

import cv2
import csv
import time
import os
import mediapipe.python.solutions.hands as mp_hands_module
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
from context_classifier import detect_context, is_cross_hands_fist
from intent_classifier import classify_intent

# Color Constants (BGR format)
COLOR_NEUTRAL = (0, 0, 255)      # Red
COLOR_READING = (0, 255, 0)      # Green
COLOR_SELECT = (0, 255, 255)     # Yellow
COLOR_DISMISS = (0, 128, 255)    # Orange
COLOR_SCROLL = (255, 255, 0)     # Cyan
COLOR_NONE = (180, 180, 180)     # Grey


def initialize_system():
    """Set up all system components once before the main loop starts."""
    hands = mp_hands_module.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)

    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/session_{int(time.time())}.csv"
    log_file = open(log_filename, 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["timestamp", "context", "intent"])

    print(f"Log file created at: {log_filename}")
    return hands, cap, log_file, csv_writer


def process_frame(frame, hands, prev_landmarks, current_context):
    """Run all three pipeline stages on one frame."""
    # Step 1: Process frame with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Step 2: Set defaults
    current_landmarks = None
    new_context = current_context
    intent = "NONE"

    # Step 3: Classify context and intent
    if result.multi_hand_landmarks:
        current_landmarks = result.multi_hand_landmarks[0]

        if len(result.multi_hand_landmarks) == 2 and is_cross_hands_fist(
            result.multi_hand_landmarks[0], result.multi_hand_landmarks[1]
        ):
            new_context = "NEUTRAL"
        else:
            new_context = detect_context(result, current_context)

        intent = classify_intent(current_landmarks, prev_landmarks, new_context)

        # Draw landmarks for all detected hands
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands_module.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # Step 4: Return results
    return frame, new_context, intent, current_landmarks


def draw_overlay(frame, context, intent):
    """Draw context label, intent label, and hint text on the video frame."""
    h, w, _ = frame.shape

    # Context box - top left
    context_color = COLOR_READING if context == "READING" else COLOR_NEUTRAL
    cv2.rectangle(frame, (0, 0), (320, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"CONTEXT: {context}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, context_color, 2)

    # Intent box - bottom left
    if intent == "SELECT":
        intent_color = COLOR_SELECT
    elif intent == "DISMISS":
        intent_color = COLOR_DISMISS
    elif intent.startswith("SCROLL"):
        intent_color = COLOR_SCROLL
    else:
        intent_color = COLOR_NONE

    cv2.rectangle(frame, (0, h - 60), (380, h), (0, 0, 0), -1)
    cv2.putText(frame, f"INTENT: {intent}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, intent_color, 2)

    # Hints - top right
    hint_color = (200, 200, 200)
    hint_font_scale = 0.5
    hint_thickness = 1
    hints = [
        "Both hands open+close = READING",
        "Cross fists = EXIT context",
        "Point index = SELECT",
        "Push palm = DISMISS",
        "Move hand up/down = SCROLL",
        "Q = Quit"
    ]
    for i, text in enumerate(hints):
        cv2.putText(frame, text, (w - 420, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, hint_font_scale, hint_color, hint_thickness)

    return frame


def log_frame_data(writer, timestamp, context, intent):
    """Write one row to the CSV log file."""
    writer.writerow([timestamp, context, intent])


def main():
    """Entry point. Runs the full pipeline loop."""
    # Step 1: Print startup messages
    print("HandIntent MVP Starting...")
    print("Show both open hands close together to enter READING context.")
    print("Cross both fists to exit context.")
    print("Press Q to quit.")

    # Step 2: Initialize system
    hands, cap, log_file, writer = initialize_system()

    # Step 3: Initialize loop variables
    current_context = "NEUTRAL"
    prev_landmarks = None

    # Step 4: Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        frame = cv2.flip(frame, 1)

        frame, current_context, intent, current_landmarks = process_frame(
            frame, hands, prev_landmarks, current_context
        )

        frame = draw_overlay(frame, current_context, intent)

        timestamp = time.strftime("%H:%M:%S")
        log_frame_data(writer, timestamp, current_context, intent)

        cv2.imshow("HandIntent MVP - Utsav", frame)

        prev_landmarks = current_landmarks

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Step 5: Cleanup
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
    print("Session ended. Log saved.")


if __name__ == "__main__":
    main()