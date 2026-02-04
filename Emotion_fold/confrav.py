import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# -------- MODEL --------
class EmotionMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -------- LOAD DATA --------
df = pd.read_csv("ravdess_egemaps.csv")

X = df.drop("emotion", axis=1).values
y = df["emotion"].values

label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# same split logic as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# -------- LOAD MODEL --------
checkpoint = torch.load(
    "emotion_model.pt",
    map_location="cpu",
    weights_only=False
)

model = EmotionMLP(X.shape[1], len(label_encoder.classes_))
model.load_state_dict(checkpoint["model_state"])
model.eval()

# -------- PREDICT --------
with torch.no_grad():
    logits = model(torch.tensor(X_test, dtype=torch.float32))
    y_pred = torch.argmax(logits, dim=1).numpy()

# -------- METRICS --------
print("\nEmotion classes:")
print(label_encoder.classes_)

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

print("\nConfusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)
