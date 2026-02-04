# ==============================
# STEP 8: CNN + BiLSTM SER MODEL
# ==============================

import torch
import torch.nn as nn

# --------------------------------
# CNN FEATURE EXTRACTOR
# --------------------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            # -------- Block 1 --------
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # pool only over frequency
            nn.Dropout(0.3),

            # -------- Block 2 --------
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.3),

            # -------- Block 3 --------
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        """
        Input:
            x -> (B, 1, 128, T)
        Output:
            x -> (B, 128, 16, T)
        """
        return self.cnn(x)


# --------------------------------
# RESHAPE CNN OUTPUT FOR BiLSTM
# --------------------------------
def reshape_for_lstm(x):
    """
    Input:
        x -> (B, C, F, T)
    Output:
        x -> (B, T, C*F)
    """
    x = x.permute(0, 3, 1, 2)     # (B, T, C, F)
    x = x.contiguous().view(x.size(0), x.size(1), -1)
    return x


# --------------------------------
# FULL SER MODEL
# --------------------------------
class SERModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.cnn = CNNFeatureExtractor()

        self.bilstm = nn.LSTM(
            input_size=2048,     # 128 channels × 16 freq bins
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Input:
            x -> (B, 1, 128, T)
        Output:
            logits -> (B, 8)
        """
        # CNN feature extraction
        x = self.cnn(x)

        # Prepare for BiLSTM
        x = reshape_for_lstm(x)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)   # (B, T, 256)

        # Temporal mean pooling
        x = torch.mean(lstm_out, dim=1)

        # Classification
        logits = self.classifier(x)
        return logits


# --------------------------------
# SANITY CHECK (RUN THIS FILE)
# --------------------------------
if __name__ == "__main__":
    model = SERModel()

    # Dummy batch: matches your real data (T = 94)
    dummy_input = torch.randn(16, 1, 128, 94)

    output = model(dummy_input)

    print("Model output shape:", output.shape)
