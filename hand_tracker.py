import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Use legacy solution which still works
import mediapipe.python.solutions.hands as mp_hands_module
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

hands = mp_hands_module.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
print("Hand tracker running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands_module.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            handedness = result.multi_handedness[idx].classification[0].label
            h, w, _ = frame.shape
            tip = hand_landmarks.landmark[8]
            x = int(tip.x * w)
            y = int(tip.y * h)

            cv2.putText(
                frame,
                f"{handedness} hand | Index: ({x},{y})",
                (10, 30 + idx * 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if handedness == "Right" else (255, 0, 0),
                2
            )

    else:
        cv2.putText(
            frame,
            "No hand detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Hand Tracker - Utsav", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()