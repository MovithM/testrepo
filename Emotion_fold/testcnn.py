# ==========================================
# STEP 10: TEST SET EVALUATION (SER)
# ==========================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import seaborn as sns

from ravcnnprog import create_dataloaders
from ravcnnbilstm import SERModel

# -------------------------
# CONFIG
# -------------------------
DATASET_PATH = "/home/roavai/Documents/Movith_progfiles/Emotion_fold/RAVDESS"
MODEL_PATH = "best_ser_model.pth"
BATCH_SIZE = 16
NUM_CLASSES = 8

EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# LOAD DATA
# -------------------------
_, _, test_loader = create_dataloaders(
    DATASET_PATH,
    batch_size=BATCH_SIZE
)

# -------------------------
# LOAD MODEL
# -------------------------
model = SERModel(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------------------------
# INFERENCE
# -------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        outputs = model(x)

        preds = torch.argmax(outputs, dim=1)

        y_true.extend(y.numpy())
        y_pred.extend(preds.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -------------------------
# METRICS
# -------------------------
accuracy = accuracy_score(y_true, y_pred)
uar = recall_score(y_true, y_pred, average="macro")

print("\n📊 TEST RESULTS")
print(f"Accuracy : {accuracy:.4f}")
print(f"UAR      : {uar:.4f}")

# -------------------------
# CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=EMOTION_LABELS,
    yticklabels=EMOTION_LABELS
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – RAVDESS (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
