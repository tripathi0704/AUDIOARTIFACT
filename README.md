# AudioArtifact v2.0 — Engineering Guide

AudioArtifact is an advanced timeline-based deepfake audio localizer. It detects and pinpoints synthetic speech segments in audio recordings using Voice Activity Detection (VAD), 50% overlapping sliding windows, dynamic MFCC derivatives (MFCC + Delta + Delta²), XGBoost continuous probability modeling, and interactive Plotly visualization.

---

## What's New in v2.0

- **Voice Activity Detection (Silero VAD)**: Automatically removes silence and ambient pauses to focus forensics strictly on active voice signals.
- **50% Overlapping Slicing**: Slices audio into 2-second windows with a 1-second stride, preventing spliced splice-point artifacts from falling between non-overlapping boundaries.
- **60-Feature Extraction**: Combines 20 MFCCs, 20 Delta coefficients (velocity), and 20 Delta-Delta coefficients (acceleration) to capture temporal frequency shifts characteristic of neural TTS/voice cloning.
- **Continuous Probability Modeling**: XGBoost calibrated with `predict_proba` to deliver second-by-second synthetic risk probabilities (0.0 to 1.0) evaluated with ROC-AUC.
- **Interactive Visual Reporting**: Streamlit frontend featuring an interactive Plotly continuous probability curve, verdict card, segment breakdown inspector, 3D voice signature spectrogram, and SQLite scan history.

---

## 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Dataset Organization

Ensure datasets are organized under `data/`:

```text
data/
├── real/    <- Genuine human voice recordings (.wav / .mp3)
└── fake/    <- AI-synthesized voice recordings (.wav / .mp3)
```

---

## 3. Feature Extraction (VAD + 50% Overlap + 60-MFCC)

```bash
# Extract features from a balanced set (default: 150 files per class)
python feature_extraction.py

# Or customize limit:
python feature_extraction.py --limit 100

# Or extract all files:
python feature_extraction.py --all
```

This generates `features.csv` with 60 feature columns (`mfcc_1..20`, `delta_1..20`, `delta2_1..20`) and `label`.

---

## 4. Train the Model

```bash
python train_model.py
```

This trains the XGBoost classifier, outputs Accuracy, Precision, Recall, F1-score, Confusion Matrix, and **ROC-AUC score**, then saves the model to `model/deepfake_detector.pkl`.

---

## 5. Launch the Application

You can launch both the backend and frontend at once using:

```bash
start.bat
```

Or start them individually in separate terminals:

**Terminal 1 (Backend - FastAPI):**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 (Frontend - Streamlit):**
```bash
streamlit run frontend/app.py
```

- **Backend API**: `http://127.0.0.1:8000` (Interactive docs at `/docs`)
- **Frontend Dashboard**: `http://localhost:8501`

---

## Project Structure

```text
AudioArtifact/
├── backend/
│   ├── main.py                  <- FastAPI async backend service (/analyze, /history)
│   ├── vad_utils.py             <- Silero VAD + energy-based silence filtering
│   └── db.py                    <- SQLite persistence & scan history
├── frontend/
│   └── app.py                   <- Streamlit dashboard with Plotly probability curve
├── data/
│   ├── real/                    <- Human audio dataset
│   └── fake/                    <- Deepfake audio dataset
├── model/
│   └── deepfake_detector.pkl    <- Trained XGBoost classifier
├── feature_extraction.py        <- 60-feature extractor with VAD and 50% overlap
├── train_model.py               <- XGBoost training with ROC-AUC evaluation
├── requirements.txt             <- Project dependencies
├── history.db                   <- SQLite forensic scan history
├── start.bat                    <- Dual-server launcher
└── README.md                    <- Documentation
```