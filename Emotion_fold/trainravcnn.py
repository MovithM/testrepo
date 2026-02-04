# ==========================================
# STEP 9: TRAINING LOOP FOR SER (RAVDESS)
# CNN + BiLSTM
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import recall_score
from tqdm import tqdm


# ---- IMPORT YOUR OWN FILES ----
from ravcnnprog import create_dataloaders   # STEP 7
from ravcnnbilstm import SERModel            # STEP 8


# -------------------------
# CONFIGURATION (LOCKED)
# -------------------------
DATASET_PATH = "/home/roavai/Documents/Movith_progfiles/Emotion_fold/RAVDESS"
BATCH_SIZE = 16
NUM_CLASSES = 8
EPOCHS = 50
PATIENCE = 7
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MODEL_PATH = "best_ser_model.pth"


# -------------------------
# DEVICE SETUP
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------
# UAR METRIC
# -------------------------
def compute_uar(y_true, y_pred):
    """
    Unweighted Average Recall (UAR)
    """
    return recall_score(y_true, y_pred, average="macro")


# -------------------------
# TRAINING FUNCTION
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    all_preds, all_labels = [], []

    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = running_loss / len(loader)
    uar = compute_uar(all_labels, all_preds)

    return avg_loss, uar


# -------------------------
# VALIDATION FUNCTION
# -------------------------
def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = running_loss / len(loader)
    uar = compute_uar(all_labels, all_preds)

    return avg_loss, uar


# -------------------------
# MAIN TRAINING LOOP
# -------------------------
def main():
    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(
        DATASET_PATH, batch_size=BATCH_SIZE
    )

    # Model
    model = SERModel(num_classes=NUM_CLASSES).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        verbose=True
    )

    best_uar = 0.0
    epochs_no_improve = 0

    # Training
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")

        train_loss, train_uar = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_uar = validate(
            model, val_loader, criterion
        )

        scheduler.step(val_loss)

        print(
            f"Train Loss: {train_loss:.4f} | Train UAR: {train_uar:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val UAR: {val_uar:.4f}"
        )

        # Early stopping + checkpoint
        if val_uar > best_uar:
            best_uar = val_uar
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("✅ Best model saved")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print("⏹ Early stopping triggered")
            break

    print(f"\n🎉 Training completed. Best Val UAR: {best_uar:.4f}")


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    main()
