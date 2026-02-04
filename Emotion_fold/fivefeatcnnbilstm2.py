import torch
import torch.nn as nn

# Import CNN from previous file
from fivefeatcnn1 import CNNFeatureExtractor
from ravcnnprog import create_dataloaders
class CNN_BiLSTM(nn.Module):
    def __init__(self, lstm_hidden=256, num_classes=5):
        super().__init__()

        # CNN feature extractor
        self.cnn = CNNFeatureExtractor()

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=4096,      # MUST match CNN output
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Classifier
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        """
        x: (batch, 1, 128, T)
        """

        # CNN
        x = self.cnn(x)
        # (batch, time, 4096)

        # BiLSTM
        lstm_out, _ = self.lstm(x)
        # (batch, time, 2*lstm_hidden)

        # TEMP pooling (will be replaced by attention)
        x = torch.mean(lstm_out, dim=1)

        # Classification
        logits = self.classifier(x)
        return logits
if __name__ == "__main__":
    root = "/home/roavai/Documents/Movith_progfiles/Emotion_fold/RAVDESS"

    train_loader, _, _ = create_dataloaders(root)

    model = CNN_BiLSTM()

    x, y = next(iter(train_loader))
    out = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
