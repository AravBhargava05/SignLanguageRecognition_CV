# Sign Language Detection (Hand Landmarks + Classifier)

## Overview
This project builds a real-time sign language (A–Z) hand-gesture classifier using webcam video. The core idea is simple:
1) collect labeled images,
2) extract hand landmark features,
3) train a lightweight classifier,
4) run live inference and overlay predictions on the camera feed.

It’s designed to be fast, understandable, and easy to extend.

---

## What this demonstrates (why it matters)
- Practical computer vision pipeline: data collection -> feature extraction -> training -> deployment
- Landmark-based perception (more robust than raw pixels for small projects)
- Real-time inference loop with bounding boxes + labels
- A clean baseline you can later upgrade (more data, better features, or a neural model)

---

## Method (high level)
- Per frame, MediaPipe Hands detects 21 hand landmarks.
- Each landmark provides normalized (x, y) coordinates in image space.
- Features are built from these coordinates and normalized to reduce sensitivity to where the hand is in the frame.
- A classical ML model (Random Forest) is trained on those features.

This gives you a strong baseline without needing to train a deep network.

---

## Files
- `collect_imgs.py`
  - Captures training images from your webcam and saves them into `data/<class_id>/`.
  - The folder name is the label (0..25).

- `create_dataset.py`
  - Reads images in `data/`, runs MediaPipe Hands, and converts each image into a feature vector.
  - Saves `data.pickle` with:
    - `data`: list of feature vectors
    - `labels`: list of integer labels (folder names)

- `train_classifier.py`
  - Loads `data.pickle`
  - Splits into train/test
  - Trains a RandomForest classifier
  - Saves model to `model.p`

- `inference_classifier.py`
  - Loads `model.p`
  - Runs webcam inference in real time
  - Draws a bounding box around the hand and prints the predicted letter

---

## How the features work
MediaPipe provides 21 landmarks (x, y). Your dataset builder normalizes the hand coordinates by subtracting the minimum x and minimum y across the landmarks.

Why:
- If you move your hand left/right/up/down, absolute coordinates change.
- Subtracting (min_x, min_y) makes the feature vector more translation-invariant.
- That makes training easier and improves generalization.

Feature shape:
- 21 landmarks * 2 coords = 42 values per frame

---

## Setup
Install dependencies:

```bash
pip install numpy opencv-python mediapipe scikit-learn

