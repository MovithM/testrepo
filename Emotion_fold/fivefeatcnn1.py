import torch
import torch.nn as nn
import torch.nn.functional as F
from ravcnnprog import (
    load_audio,
    extract_mel,
    EMOTION_MAP,
    RAVDESSDataset,
    split_by_speaker,
    create_dataloaders
)

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))  # ↓ freq only
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))  # ↓ freq only
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: (batch, 1, 128, T)
        """
        x = self.conv_block1(x)  # (B, 32, 64, T)
        x = self.conv_block2(x)  # (B, 64, 32, T)
        x = self.conv_block3(x)  # (B, 128, 32, T)

        # Prepare for BiLSTM:
        # (B, C, F, T) → (B, T, C*F)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), -1)

        return x
if __name__ == "__main__":
    root = "/home/roavai/Documents/Movith_progfiles/Emotion_fold/RAVDESS"

    train_loader, val_loader, test_loader = create_dataloaders(root)

    model = CNNFeatureExtractor()

    x, _ = next(iter(train_loader))
    out = model(x)

    print("CNN output shape:", out.shape)
