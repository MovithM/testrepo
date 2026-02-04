# =========================
# RAVDESS DATASET PIPELINE
# CNN + BiLSTM READY
# =========================

import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# -------------------------
# FIXED PARAMETERS (LOCKED)
# -------------------------
SAMPLE_RATE = 16000
DURATION = 3  # seconds
NUM_SAMPLES = SAMPLE_RATE * DURATION
N_MELS = 128
BATCH_SIZE = 16

# -------------------------
# EMOTION LABEL MAPPING
# -------------------------
EMOTION_MAP = {
    "01": 0,  # Neutral  → Neutral
    "02": 0,  # Calm     → Neutral
    "03": 1,  # Happy    → Happy
    "04": 2,  # Sad      → Sad
    "05": 3,  # Angry    → Angry
    "07": 3,  # Disgust  → Angry
    "06": 4,  # Fearful  → Surprised
    "08": 4   # Surprised→ Surprised
}

# -------------------------
# AUDIO PREPROCESSING
# -------------------------
def load_audio(path):
    waveform, sr = torchaudio.load(path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if required
    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # Pad or truncate to fixed length
    if waveform.shape[1] < NUM_SAMPLES:
        pad_len = NUM_SAMPLES - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :NUM_SAMPLES]

    return waveform


# -------------------------
# MEL-SPECTROGRAM
# -------------------------
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=2048,
    hop_length=512,
    n_mels=N_MELS
)

amplitude_to_db = T.AmplitudeToDB()


def extract_mel(waveform):
    mel = mel_transform(waveform)     # (1, 128, T)
    mel = amplitude_to_db(mel)
    return mel


# -------------------------
# SPEAKER SPLIT UTILITIES
# -------------------------
def get_speaker_id(path):
    # Actor_01 -> 1
    return int(os.path.basename(os.path.dirname(path)).split("_")[1])


def split_by_speaker(file_list):
    train, val, test = [], [], []

    for path in file_list:
        speaker = get_speaker_id(path)

        if speaker <= 18:
            train.append(path)
        elif speaker <= 21:
            val.append(path)
        else:
            test.append(path)

    return train, val, test


# -------------------------
# DATASET CLASS
# -------------------------
class RAVDESSDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]

        # Extract emotion label from filename
        filename = os.path.basename(path)
        emotion_code = filename.split("-")[2]
        label = EMOTION_MAP[emotion_code]

        # Audio -> Mel Spectrogram
        waveform = load_audio(path)
        mel = extract_mel(waveform)

        return mel, label


# -------------------------
# DATALOADER CREATION
# -------------------------
def create_dataloaders(root_dir, batch_size=BATCH_SIZE):
    all_files = []

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".wav"):
                all_files.append(os.path.join(root, f))

    train_files, val_files, test_files = split_by_speaker(all_files)

    train_ds = RAVDESSDataset(train_files)
    val_ds = RAVDESSDataset(val_files)
    test_ds = RAVDESSDataset(test_files)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# -------------------------
# SANITY CHECK (RUN ONCE)
# -------------------------
if __name__ == "__main__":
    root = "/home/roavai/Documents/Movith_progfiles/Emotion_fold/RAVDESS"

    train_loader, val_loader, test_loader = create_dataloaders(root)

    x, y = next(iter(train_loader))
    print("Mel shape:", x.shape)
    print("Labels shape:", y.shape)
    from collections import Counter
    import torch

    # Collect all training labels
    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.tolist())

    label_counts = Counter(all_labels)
    print("Label counts:", label_counts)

    num_classes = 5
    total_samples = sum(label_counts.values())

    # Inverse frequency weighting
    class_weights = []
    for i in range(num_classes):
        class_weights.append(total_samples / label_counts[i])

    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print("Class weights:", class_weights)

