# train_classifier.py  (drop-in fix)
import os, pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_HERE  = os.path.join(BASE, "data.pickle")
DATA_UPONE = os.path.join(os.path.dirname(BASE), "data.pickle")
DATA_PATH  = DATA_HERE if os.path.exists(DATA_HERE) else DATA_UPONE

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"data.pickle not found.\nLooked in:\n  {DATA_HERE}\n  {DATA_UPONE}\n"
        "Move/copy data.pickle to one of those and retry."
    )

print("Using dataset:", DATA_PATH)
with open(DATA_PATH, "rb") as f:
    d = pickle.load(f)

X = np.asarray(d["data"], dtype=float)
y = np.asarray(d["labels"])

print(f"Loaded {len(X)} samples across {len(set(y))} classes.")
if len(X) == 0:
    raise RuntimeError("Dataset is empty. Rebuild with create_dataset.py.")

# Train/test split (fallback if stratify can’t be used)
try:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
except ValueError:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print("Training RandomForest...")
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(Xtr, ytr)

print("Evaluating...")
pred = clf.predict(Xte)
acc = accuracy_score(yte, pred)
print(f"Accuracy: {acc*100:.2f}%")

MODEL_PATH = os.path.join(BASE, "model.p")
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": clf}, f)
print("Saved model to:", MODEL_PATH)
