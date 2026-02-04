import sounddevice as sd
import numpy as np
import queue
import time
import torch
import torch.nn.functional as F
import soundfile as sf
import uuid
import subprocess
import os

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024        # ~64 ms
WINDOW_SEC = 2.0
ENERGY_THRESH = 0.003

OPENSMILE_DIR = "/home/movith/opensmile/build/progsrc/smilextract"
CONF_PATH = "../../../config/egemaps/v02/eGeMAPSv02.conf"

# ---------------- LOAD MODEL ----------------
checkpoint = torch.load(
    "emotion_model.pt",
    map_location="cpu",
    weights_only=False
)

INPUT_DIM = checkpoint.get("input_dim", 88)
NUM_CLASSES = len(checkpoint["label_encoder"].classes_)

model = EmotionMLP(INPUT_DIM, NUM_CLASSES)
model.load_state_dict(checkpoint["model_state"])
model.eval()

scaler = checkpoint["scaler"]
label_encoder = checkpoint["label_encoder"]


# ---------------- AUDIO BUFFER ----------------
audio_q = queue.Queue()
audio_buffer = np.zeros(0, dtype=np.float32)

# ---------------- CALLBACK ----------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(indata[:, 0].copy())

# ---------------- FEATURE EXTRACTION ----------------
def extract_egemaps(audio):
    tmp_wav = f"/tmp/{uuid.uuid4().hex}.wav"
    tmp_arff = f"/tmp/{uuid.uuid4().hex}.arff"

    audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    sf.write(tmp_wav, audio_int16, SAMPLE_RATE)

    subprocess.run(
        ["./SMILExtract", "-C", CONF_PATH, "-I", tmp_wav, "-O", tmp_arff],
        cwd=OPENSMILE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    feats = None
    if os.path.exists(tmp_arff):
        with open(tmp_arff) as f:
            data = False
            for line in f:
                if line.lower().strip() == "@data":
                    data = True
                    continue
                if data and line.strip():
                    vals = line.split(",")[1:-1]
                    feats = np.array(vals, dtype=np.float32)
                    break

    os.remove(tmp_wav)
    if os.path.exists(tmp_arff):
        os.remove(tmp_arff)

    return feats

# ---------------- REAL-TIME LOOP ----------------
print("🎙️ Listening (Ctrl+C to stop)...")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=1,
    callback=audio_callback
):
    try:
        while True:
            while not audio_q.empty():
                audio_buffer = np.concatenate([audio_buffer, audio_q.get()])

            if len(audio_buffer) >= int(WINDOW_SEC * SAMPLE_RATE):
                window = audio_buffer[:int(WINDOW_SEC * SAMPLE_RATE)]
                audio_buffer = audio_buffer[int(0.5 * SAMPLE_RATE):]  # hop

                rms = np.sqrt(np.mean(window ** 2))
                if rms < ENERGY_THRESH:
                    continue

                feats = extract_egemaps(window)
                if feats is None:
                    continue

                feats = scaler.transform([feats])
                x = torch.tensor(feats, dtype=torch.float32)

                with torch.no_grad():
                    probs = F.softmax(model(x), dim=1)[0]

                pred = torch.argmax(probs).item()
                emotion = label_encoder.inverse_transform([pred])[0]
                confidence = probs[pred].item()

                print(f"🎭 {emotion} ({confidence*100:.1f}%)")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")
