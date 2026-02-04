# ==========================================
# CNN + BiLSTM + ATTENTION for SER
# ==========================================

import torch
import torch.nn as nn

# --------------------------------
# CNN FEATURE EXTRACTOR (unchanged)
# --------------------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.cnn(x)


# --------------------------------
# RESHAPE FOR LSTM
# --------------------------------
def reshape_for_lstm(x):
    x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
    x = x.contiguous().view(x.size(0), x.size(1), -1)
    return x


# --------------------------------
# TEMPORAL ATTENTION MODULE
# --------------------------------
class TemporalAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x: (B, T, D)
        returns: (B, D)
        """
        scores = self.attention(x)        # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context


# --------------------------------
# FULL SER MODEL WITH ATTENTION
# --------------------------------
class SERModelAttention(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.cnn = CNNFeatureExtractor()

        self.bilstm = nn.LSTM(
            input_size=2048,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.attention = TemporalAttention(input_dim=256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = reshape_for_lstm(x)

        lstm_out, _ = self.bilstm(x)   # (B, T, 256)

        x = self.attention(lstm_out)  # (B, 256)

        logits = self.classifier(x)
        return logits


# --------------------------------
# SANITY CHECK
# --------------------------------
if __name__ == "__main__":
    model = SERModelAttention()
    dummy = torch.randn(16, 1, 128, 94)
    out = model(dummy)
    print("Output shape:", out.shape)
