import subprocess
import tempfile
import numpy as np
import torch
import torch.nn as nn

# -------- MODEL DEFINITION --------
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

# -------- CONFIG --------
OPENSMILE_BIN = "/home/movith/opensmile/build/progsrc/smilextract/SMILExtract"
EGEMAPS_CONF = "/home/movith/opensmile/config/egemaps/v02/eGeMAPSv02.conf"


INPUT_DIM = 88
NUM_CLASSES = 8

# -------- LOAD CHECKPOINT --------
checkpoint = torch.load(
    "emotion_model.pt",
    map_location="cpu",
    weights_only=False
)

model = EmotionMLP(INPUT_DIM, NUM_CLASSES)
model.load_state_dict(checkpoint["model_state"])
model.eval()

scaler = checkpoint["scaler"]
label_encoder = checkpoint["label_encoder"]

def extract_egemaps(wav_path):
    out_name = "tmp_egemaps.arff"

    subprocess.run(
        [
            "./SMILExtract",
            "-C", "../../../config/egemaps/v02/eGeMAPSv02.conf",
            "-I", wav_path,
            "-O", out_name
        ],
        cwd="/home/movith/opensmile/build/progsrc/smilextract",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    out_path = "/home/movith/opensmile/build/progsrc/smilextract/" + out_name

    with open(out_path, "r") as f:
        data_section = False
        for line in f:
            line = line.strip()

            if line.lower() == "@data":
                data_section = True
                continue

            if data_section and line and not line.startswith("%"):
                values = line.split(",")[1:-1]  # 🔥 FIX HERE
                features = np.array(values, dtype=np.float32)

                if features.shape[0] != 88:
                    raise RuntimeError(
                        f"Expected 88 features, got {features.shape[0]}"
                    )

                return features

    raise RuntimeError("❌ No features extracted")

# -------- PREDICT --------
import torch.nn.functional as F

def predict_emotion(wav_path):
    features = extract_egemaps(wav_path)
    features = scaler.transform([features])
    x = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)

    emotion = label_encoder.inverse_transform([pred])[0]
    confidence = probs[pred]

    return emotion, confidence, probs


# -------- RUN --------
if __name__ == "__main__":
    wav_path = "/home/movith/Datasets/RAVDESS/Actor_05/03-01-03-02-01-01-05.wav"
    print("Using WAV:", wav_path)
    emotion, confidence, probs = predict_emotion(wav_path)
    print(f"\n🎭 Predicted Emotion: {emotion} ({confidence*100:.1f}%)")

