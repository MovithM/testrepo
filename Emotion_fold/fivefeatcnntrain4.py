import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.metrics import recall_score

from ravcnnprog import create_dataloaders
from fivefeatcnnatten3 import CNN_BiLSTM_Attention
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

EPOCHS = 20
LR = 1e-4
BATCH_SIZE = 16
NUM_CLASSES = 5

root = "/home/roavai/Documents/Movith_progfiles/Emotion_fold/RAVDESS"
train_loader, val_loader, test_loader = create_dataloaders(root)

all_labels = []
for _, y in train_loader:
    all_labels.extend(y.tolist())

label_counts = Counter(all_labels)
total = sum(label_counts.values())

class_weights = torch.tensor(
    [total / label_counts[i] for i in range(NUM_CLASSES)],
    dtype=torch.float
).to(device)

print("Class weights:", class_weights)

model = CNN_BiLSTM_Attention(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)


def compute_uar(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            outputs, _ = model(x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # UAR = macro recall
    uar = recall_score(all_labels, all_preds, average="macro")
    return uar


for epoch in range(EPOCHS):
    # -------- TRAIN --------
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs, _ = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            outputs, _ = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_uar = compute_uar(model, val_loader, device)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
        f"Val UAR: {val_uar:.4f}"
    )

    # -------- SAVE BEST MODEL --------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print("✅ Best model saved")

