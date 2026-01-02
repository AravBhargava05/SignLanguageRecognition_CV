# inference_classifier.py (fixed)
import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# --- Paths anchored to this file ---
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model.p")

# Load model.p robustly
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"model.p not found at: {MODEL_PATH}")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)["model"]

# --- Camera (try 0, then 1) ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("No camera found. Try a different index or check permissions.")

# --- MediaPipe Hands (live video) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# A–Z mapping (0->A ... 25->Z)
labels_dict = {i: chr(ord('A') + i) for i in range(26)}

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    H, W = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux, x_, y_ = [], [], []

        # Draw landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Collect normalized features (first hand is enough if you trained on one)
        for hand_landmarks in results.multi_hand_landmarks[:1]:
            for lm in hand_landmarks.landmark:
                x_.append(lm.x); y_.append(lm.y)
            minx, miny = min(x_), min(y_)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - minx); data_aux.append(lm.y - miny)

        # Predict only if we actually built a feature vector
        if data_aux:
            pred = model.predict([np.asarray(data_aux)])
            # Your training labels are directory names "0","1",... so prediction is likely a string
            try:
                idx = int(pred[0])
            except (ValueError, TypeError):
                # If you've changed labels to letters, map them back as needed
                # For now, just display the raw prediction
                idx = None

            # Bounding box (clamped)
            x1 = max(0, int(min(x_) * W) - 10)
            y1 = max(0, int(min(y_) * H) - 10)
            x2 = min(W - 1, int(max(x_) * W) + 10)
            y2 = min(H - 1, int(max(y_) * H) + 10)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            label = labels_dict[idx] if idx is not None and idx in labels_dict else str(pred[0])
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
