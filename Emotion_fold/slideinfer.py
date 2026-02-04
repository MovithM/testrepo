import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import subprocess
import os
import uuid
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

# -------- CONFIG --------
OPENSMILE_DIR = "/home/movith/opensmile/build/progsrc/smilextract"
OPENSMILE_BIN = "./SMILExtract"
CONF_PATH = "../../../config/egemaps/v02/eGeMAPSv02.conf"

WINDOW_SEC = 2.0
HOP_SEC = 0.5
EXPECTED_FEATS = 88

# -------- LOAD MODEL --------
checkpoint = torch.load(
    "emotion_model.pt",
    map_location="cpu",
    weights_only=False
)

model = EmotionMLP(EXPECTED_FEATS, len(checkpoint["label_encoder"].classes_))
model.load_state_dict(checkpoint["model_state"])
model.eval()

scaler = checkpoint["scaler"]
label_encoder = checkpoint["label_encoder"]

# -------- FEATURE EXTRACTION --------

    
def extract_egemaps_segment(wav_path):
    arff_name = f"tmp_{uuid.uuid4().hex}.arff"

    subprocess.run(
        [
            "./SMILExtract",
            "-C", "../../../config/egemaps/v02/eGeMAPSv02.conf",
            "-I", wav_path,
            "-O", arff_name
        ],
        cwd="/home/movith/opensmile/build/progsrc/smilextract",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    arff_path = "/home/movith/opensmile/build/progsrc/smilextract/" + arff_name

    if not os.path.exists(arff_path):
        return None   # openSMILE skipped this window

    with open(arff_path) as f:
        data = False
        for line in f:
            line = line.strip()
            if line.lower() == "@data":
                data = True
                continue
            if data and line and not line.startswith("%"):
                vals = line.split(",")[1:-1]
                feats = np.array(vals, dtype=np.float32)

                os.remove(arff_path)  # cleanup

                if feats.shape[0] == 88:
                    return feats

    os.remove(arff_path)
    return None

#---------vad----------
def vad_segments(audio, sr, energy_thresh=0.003, min_len_sec=0.4):
    frame_len = int(0.025 * sr)   # 25 ms
    hop_len = int(0.010 * sr)     # 10 ms

    voiced = []
    start = None

    for i in range(0, len(audio) - frame_len, hop_len):
        frame = audio[i:i+frame_len]
        rms = np.sqrt(np.mean(frame**2))

        if rms > energy_thresh:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i
                if (end - start) / sr >= min_len_sec:
                    voiced.append(audio[start:end])
                start = None

    # 🔥 IMPORTANT: handle speech that goes till end
    if start is not None:
        end = len(audio)
        if (end - start) / sr >= min_len_sec:
            voiced.append(audio[start:end])

    return voiced




# -------- SLIDING WINDOW INFERENCE --------
def predict_emotion_sliding(wav_path):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    win_len = int(WINDOW_SEC * sr)
    hop_len = int(HOP_SEC * sr)

    probs_list = []

    voiced_segments = vad_segments(audio, sr)

    if len(voiced_segments) == 0:
        voiced_segments = [audio]


    for seg in voiced_segments:
        if len(seg) < win_len:
            continue

        for start in range(0, len(seg) - win_len + 1, hop_len):
            segment = seg[start:start + win_len]

            peak = np.max(np.abs(segment))
            if peak < 1e-6:
                continue

            segment_int16 = np.int16(segment / peak * 32767)

            tmp_wav = f"/tmp/tmp_{uuid.uuid4().hex}.wav"
            sf.write(tmp_wav, segment_int16, sr, subtype="PCM_16")

            print(f"Window RMS energy: {np.sqrt(np.mean(segment**2)):.6f}")

            feats = extract_egemaps_segment(tmp_wav)
            os.remove(tmp_wav)

            if feats is None:
                continue

            feats = scaler.transform([feats])
            x = torch.tensor(feats, dtype=torch.float32)

            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1).numpy()[0]

            probs_list.append(probs)


    print(f"Valid windows used: {len(probs_list)}")
    if len(probs_list) == 0:
        print("⚠️ No valid windows — falling back to full utterance")

        peak = np.max(np.abs(audio))
        if peak < 1e-6:
            return "uncertain", 0.0

        audio_int16 = np.int16(audio / peak * 32767)
        tmp_wav = "/tmp/tmp_full.wav"
        sf.write(tmp_wav, audio_int16, sr, subtype="PCM_16")

        feats = extract_egemaps_segment(tmp_wav)
        os.remove(tmp_wav)

        if feats is None:
            return "uncertain", 0.0

        feats = scaler.transform([feats])
        x = torch.tensor(feats, dtype=torch.float32)

        with torch.no_grad():
            probs = F.softmax(model(x), dim=1).numpy()[0]

        pred = np.argmax(probs)
        return label_encoder.inverse_transform([pred])[0], probs[pred]
    mean_probs = np.mean(probs_list, axis=0)
    pred = np.argmax(mean_probs)
    confidence = mean_probs[pred]

    emotion = label_encoder.inverse_transform([pred])[0]
    return emotion, confidence

# -------- RUN --------
if __name__ == "__main__":
    wav_path = "/home/movith/Documents/tmp_segment.wav"
    emotion, confidence = predict_emotion_sliding(wav_path)

    if confidence < 0.2:
        emotion = "uncertain"

    print(f"\n🎭 Predicted Emotion: {emotion} ({confidence*100:.1f}%)")
