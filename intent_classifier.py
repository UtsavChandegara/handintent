# intent_classifier.py (Version 2)
# Stage 3 of HandIntent MVP Pipeline - ML-based intent classification
# Author: Utsav

from typing import Optional
import torch
import torch.nn as nn
import numpy as np

# --- Constants (Must match train_bigru.py) ---
INTENTS = ["SELECT", "DISMISS", "SCROLL_UP", "SCROLL_DOWN"]
FRAMES_PER_SAMPLE = 30
LANDMARKS_PER_FRAME = 21
COORDS_PER_LANDMARK = 3
INPUT_SIZE = LANDMARKS_PER_FRAME * COORDS_PER_LANDMARK  # 63
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = len(INTENTS)
MODEL_PATH = "models/bigru_intent.pth"
SEQUENCE_LENGTH = 30

# --- Model Definition (Copied from train_bigru.py) ---
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

# --- Global State ---
_landmark_buffer = []
_model = None

# --- Core Functions ---
def load_model():
    """Load trained BiGRU model from disk. Called once at startup."""
    model = BiGRUClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        print(f"BiGRU model loaded from {MODEL_PATH}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run train_bigru.py first.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def normalize_landmarks(hand_landmarks):
    """Make coordinates wrist-relative and flatten them."""
    landmarks = hand_landmarks.landmark
    wrist = landmarks[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    flat_coords = []
    for landmark in landmarks:
        flat_coords.append(landmark.x - wrist_x)
        flat_coords.append(landmark.y - wrist_y)
        flat_coords.append(landmark.z - wrist_z)
    return flat_coords

def update_buffer(hand_landmarks):
    """Add current frame landmarks to the rolling buffer."""
    global _landmark_buffer
    frame_features = normalize_landmarks(hand_landmarks)
    _landmark_buffer.append(frame_features)
    if len(_landmark_buffer) > SEQUENCE_LENGTH:
        _landmark_buffer.pop(0)

def buffer_to_tensor():
    """Convert current buffer contents to a tensor for model input."""
    if len(_landmark_buffer) < SEQUENCE_LENGTH:
        return None
    
    np_buffer = np.array(_landmark_buffer, dtype=np.float32)
    tensor = torch.from_numpy(np_buffer)
    return tensor.unsqueeze(0) # Add batch dimension -> (1, 30, 63)

def predict_intent():
    """Run model inference on the current buffer."""
    global _model
    if _model is None:
        return "NONE"

    input_tensor = buffer_to_tensor()
    if input_tensor is None:
        return "NONE"

    with torch.no_grad():
        output = _model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities.max().item()
        predicted_class_index = torch.argmax(output, dim=1).item()

        if confidence < 0.6:
            return "NONE"
        
        return INTENTS[predicted_class_index]

# --- Main Interface Function ---
def classify_intent(
    hand_landmarks: Optional[object],
    prev_landmarks: Optional[object],
    context: str
) -> str:
    """
    Main intent classification function. Uses the BiGRU model for inference.
    The signature is kept identical to the rule-based version for compatibility.
    """
    if not hand_landmarks:
        # If no hand is detected, clear the buffer to prevent stale predictions
        global _landmark_buffer
        _landmark_buffer.clear()
        return "NONE"

    if context == "NEUTRAL":
        return "NONE"

    if context == "READING":
        update_buffer(hand_landmarks)
        return predict_intent()

    return "NONE"

# --- Module Initialization ---
_model = load_model()

if __name__ == "__main__":
    print("intent_classifier.py v2 loaded successfully")
    print("Model ready:", _model is not None)
    if _model:
        print("classify_intent(None, None, 'NEUTRAL') →",
            classify_intent(None, None, "NEUTRAL"))