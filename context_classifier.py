# context_classifier.py
# Stage 2 of HandIntent MVP Pipeline
# Detects context changes based on two-hand gestures.
# Author: Utsav

def is_hand_open(hand_lm):
    """Helper to check if a single hand is open (at least 3 fingers extended)."""
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    bases = [5, 9, 13, 17] # Index, Middle, Ring, Pinky bases
    extended_fingers = 0
    for tip_idx, base_idx in zip(tips, bases):
        # Finger is extended if tip is higher (smaller y) than base
        if hand_lm.landmark[tip_idx].y < hand_lm.landmark[base_idx].y:
            extended_fingers += 1
    return extended_fingers >= 3

def detect_context(result, current_context="NEUTRAL"):
    """
    Detects if the context should change to 'READING'.
    The trigger is showing two open hands close together.
    """
    # Only check for context entry if we are in NEUTRAL
    if current_context == "NEUTRAL":
        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
            landmarks1 = result.multi_hand_landmarks[0]
            landmarks2 = result.multi_hand_landmarks[1]

            # Check if both hands are open and their wrists are close
            if is_hand_open(landmarks1) and is_hand_open(landmarks2):
                x1 = landmarks1.landmark[0].x
                x2 = landmarks2.landmark[0].x
                if abs(x1 - x2) < 0.2: # Threshold for wrists being close
                    return "READING"

    # If conditions are not met, or we are already in a context, do not change it.
    return current_context

def is_cross_hands_fist(landmarks1, landmarks2):
    tips = [8, 12, 16, 20]
    bases = [5, 9, 13, 17]

    def is_fist(hand_lm):
        curled = 0
        for tip, base in zip(tips, bases):
            if hand_lm.landmark[tip].y > hand_lm.landmark[base].y:
                curled += 1
        return curled >= 3

    x1 = landmarks1.landmark[0].x
    x2 = landmarks2.landmark[0].x
    wrists_close = abs(x1 - x2) < 0.15

    return is_fist(landmarks1) and is_fist(landmarks2) and wrists_close