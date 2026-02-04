import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
CSV_PATH = "ravdess_egemaps.csv"
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

# ---------------- LOAD DATA ----------------
df = pd.read_csv(CSV_PATH)

X = df.drop(columns=["emotion"]).values.astype(np.float32)
y = df["emotion"].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(label_encoder.classes_)
print("Emotion classes:", list(label_encoder.classes_))

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------- DATASET ----------------
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = EmotionDataset(X_train, y_train)
val_ds = EmotionDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ---------------- MODEL ----------------
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

model = EmotionMLP(input_dim=X.shape[1], num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAINING LOOP ----------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.3f} | Val Acc: {acc:.2f}%")

# ---------------- SAVE EVERYTHING ----------------
torch.save({
    "model_state": model.state_dict(),
    "scaler": scaler,
    "label_encoder": label_encoder,
    "input_dim": X.shape[1]
}, "emotion_model.pt")

print("✅ Model saved as emotion_model.pt")
