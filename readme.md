# Speech Emotion Recognition (SER) Module

This folder contains all code related to the Speech Emotion Recognition (SER) system developed using deep learning techniques.

---

## 📌 Project Overview
The goal of this module is to recognize human emotions from speech signals using spectrogram-based features and neural networks.

The system is trained on the RAVDESS dataset and focuses on a reduced emotion set for better robustness.

---

## 🎯 Target Emotions
The original 8 emotions are mapped to 5 classes:

- Neutral (Neutral + Calm)
- Happy
- Sad
- Angry (Angry + Disgust)
- Surprised (Surprise + Fear)

---

## 🧠 Models Implemented

- CNN-based feature extractor  
- CNN + BiLSTM  
- CNN + BiLSTM with Attention  

---

## 📂 Important Files

| File | Description |
|-----|------------|
| `fivefeatcnn1.py` | CNN baseline model |
| `fivefeatcnnbilstm2.py` | CNN + BiLSTM model |
| `fivefeatcnnatten3.py` | CNN + BiLSTM + Attention |
| `trainravcnn.py` | Training script |
| `ravfeatprog.py` | Feature extraction |
| `slideinfer.py` | Inference script |
| `ravcnnbilstm_attention.py` | Attention-based model |

---

## 📊 Features Used
- Log-Mel Spectrograms  
- eGeMAPS (88 features)

---

## 📈 Evaluation Metric
- Unweighted Average Recall (UAR)

---

## ⚠️ Note
Audio files, datasets, and trained model weights are excluded from this repository.  
Please download the dataset separately.

---

## 👨‍💻 Author
Movith
