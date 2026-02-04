import torch
import torch.nn as nn

from fivefeatcnn1 import CNNFeatureExtractor
from ravcnnprog import create_dataloaders


class TemporalAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        x: (batch, time, features)
        """
        # Compute attention scores
        scores = self.attn(x)                  # (batch, time, 1)
        weights = torch.softmax(scores, dim=1) # (batch, time, 1)

        # Weighted sum
        context = torch.sum(weights * x, dim=1)  # (batch, features)

        return context, weights
class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, lstm_hidden=256, num_classes=5):
        super().__init__()

        # CNN
        self.cnn = CNNFeatureExtractor()

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=4096,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Attention
        self.attention = TemporalAttention(input_dim=lstm_hidden * 2)

        # Classifier
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        """
        x: (batch, 1, 128, T)
        """
        # CNN
        x = self.cnn(x)               # (batch, time, 4096)

        # BiLSTM
        lstm_out, _ = self.lstm(x)    # (batch, time, 2*lstm_hidden)

        # Attention
        context, attn_weights = self.attention(lstm_out)
        # context: (batch, 2*lstm_hidden)

        # Classification
        logits = self.classifier(context)

        return logits, attn_weights
if __name__ == "__main__":
    root = "/home/roavai/Documents/Movith_progfiles/Emotion_fold/RAVDESS"

    train_loader, _, _ = create_dataloaders(root)

    model = CNN_BiLSTM_Attention()

    x, y = next(iter(train_loader))
    out, attn = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Attention shape:", attn.shape)
