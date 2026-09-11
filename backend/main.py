"""
main.py
-------
Purpose : FastAPI asynchronous backend for AudioArtifact v2.0.

Flow on every /analyze request:
  1. Receive uploaded audio file (.mp3 / .wav)
  2. Voice Activity Detection (Silero VAD) to filter out non-speech silence
  3. Slice speech with 50% overlapping windows (2.0s window, 1.0s stride)
  4. Extract 60-feature vector per segment (MFCC + Delta + Delta-Delta)
  5. Predict continuous synthetic probabilities via model.predict_proba()
  6. Calculate metrics: fake_ratio, highest_risk_segment, verdict
  7. Generate lightweight Mel-Spectrogram signature payload
  8. Persist analysis run to SQLite history
  9. Return structured, timestamped forensic JSON to the frontend
"""

import os
import sys
import tempfile
from contextlib import asynccontextmanager
import joblib
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# allow importing local modules
sys.path.append(os.path.dirname(__file__))
from db import init_db, save_result, get_history  # noqa: E402
from vad_utils import filter_speech_vad  # noqa: E402

# ---------------- CONFIG ----------------
WINDOW_SEC = 2.0              # 2-second analysis window
STRIDE_SEC = 1.0              # 1-second stride -> 50% overlap
N_MFCC = 20
SAMPLE_RATE = 16000
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "deepfake_detector.pkl")
LABEL_NAMES = {0: "Human", 1: "AI Fake"}
# -----------------------------------------

model = None


def get_model():
    """Lazily load detection model and initialize database if needed."""
    global model
    init_db()
    if model is not None:
        return model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"[Backend] Deepfake detector model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"[Backend] Error loading model: {e}")
    else:
        print(f"[Backend] WARNING: No model found at {MODEL_PATH}. Train one with train_model.py")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(title="AudioArtifact API v2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_features(segment: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extract 60-feature vector: 20 MFCC + 20 Delta + 20 Delta2 (mean across time).
    Matches feature_extraction.py v2.0 specification.
    """
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    return np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1),
        np.mean(delta2_mfcc, axis=1)
    ])


def slice_overlapping(y: np.ndarray, sr: int = SAMPLE_RATE, window_sec: float = WINDOW_SEC, stride_sec: float = STRIDE_SEC):
    """
    Slice audio into 50% overlapping windows: 2.0s window with 1.0s stride.
    Returns:
        List of tuples: (start_time_sec, end_time_sec, segment_waveform)
    """
    win_len = int(window_sec * sr)
    stride_len = int(stride_sec * sr)
    segments = []

    if len(y) < win_len:
        if len(y) >= stride_len:
            padded = np.pad(y, (0, win_len - len(y)), mode="constant")
            segments.append((0.0, window_sec, padded))
        return segments

    for start in range(0, len(y) - win_len + 1, stride_len):
        end = start + win_len
        segments.append((round(start / sr, 3), round(end / sr, 3), y[start:end]))

    return segments


def analyze_audio_data(content: bytes, original_filename: str = "upload.wav") -> dict:
    """
    Core forensic analysis engine. Can be called directly or via FastAPI.
    """
    clf = get_model()
    if clf is None:
        return {"error": "Detection model not loaded. Please train the model using train_model.py first."}

    suffix = os.path.splitext(original_filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return {"error": f"Failed to decode audio file: {e}"}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    raw_duration = round(len(y) / sr, 2)
    sr_int: int = int(sr)

    # 1. Voice Activity Detection (silence removal)
    y_speech, speech_intervals = filter_speech_vad(y, sr_int)
    if len(y_speech) < int(SAMPLE_RATE * 1.0):
        # If VAD was too restrictive, fall back to raw waveform
        y_speech = y

    # 2. Overlapping sliding window (2s window, 1s stride)
    segments = slice_overlapping(y_speech, sr_int, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC)

    if not segments:
        return {
            "error": f"Audio clip is too short ({raw_duration}s). Minimum recommended length is 2 seconds."
        }

    results = []
    fake_count = 0
    highest_risk = {"segment": 0, "start": 0.0, "end": 0.0, "fake_probability": 0.0, "confidence": 0.0}

    # 3. Predict continuous probabilities for each segment
    for i, (start_t, end_t, seg) in enumerate(segments):
        feat = extract_features(seg, sr_int).reshape(1, -1)
        probas = clf.predict_proba(feat)[0]  # [prob_human, prob_fake]

        fake_prob = float(probas[1]) if len(probas) > 1 else float(clf.predict(feat)[0])
        human_prob = float(probas[0]) if len(probas) > 1 else (1.0 - fake_prob)

        pred = 1 if fake_prob >= 0.50 else 0
        confidence = round((fake_prob if pred == 1 else human_prob) * 100, 2)

        if pred == 1:
            fake_count += 1

        if fake_prob > highest_risk["fake_probability"]:
            highest_risk = {
                "segment": i,
                "start": start_t,
                "end": end_t,
                "fake_probability": round(fake_prob, 4),
                "confidence": confidence,
            }

        results.append({
            "segment": i,
            "start": start_t,
            "end": end_t,
            "label": LABEL_NAMES[pred],
            "label_code": pred,
            "fake_probability": round(fake_prob, 4),
            "confidence": confidence,
        })

    # Summary forensic calculations
    total_analyzed_duration = round(segments[-1][1], 2)
    fake_ratio = round((fake_count / len(segments)) * 100, 1)
    fake_seconds = round(fake_count * STRIDE_SEC, 2)

    if fake_ratio == 0:
        verdict = "Authentic Human Voice"
    elif fake_ratio >= 80.0:
        verdict = "Fully Synthetic Voice Detected"
    else:
        verdict = "Spliced Audio Detected"

    # 4. Mel-Spectrogram voice signature payload for visual reporting
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    max_frames = 100
    if mel_db.shape[1] > max_frames:
        idx = np.linspace(0, mel_db.shape[1] - 1, max_frames).astype(int)
        mel_db = mel_db[:, idx]

    time_axis = np.linspace(0, raw_duration, mel_db.shape[1]).round(2).tolist()
    mel_axis = list(range(mel_db.shape[0]))

    response = {
        "filename": original_filename,
        "total_duration": raw_duration,
        "analyzed_duration": total_analyzed_duration,
        "segments_count": len(segments),
        "fake_seconds": fake_seconds,
        "fake_ratio": fake_ratio,
        "verdict": verdict,
        "highest_risk_segment": highest_risk,
        "segments": results,
        "spectrogram": {
            "z": mel_db.round(2).tolist(),
            "time": time_axis,
            "mel": mel_axis,
        },
    }

    # 5. Persist run to SQLite history
    try:
        save_result(original_filename, response)
    except Exception as e:
        print(f"[Backend] Warning: could not save to history.db: {e}")

    return response


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    return analyze_audio_data(content, file.filename or "upload.wav")


@app.get("/history")
def history(limit: int = 20):
    return get_history(limit)


@app.get("/")
def root():
    return {
        "status": "AudioArtifact backend running v2.0",
        "model_loaded": model is not None,
        "vad": "Silero VAD active",
        "window_sec": WINDOW_SEC,
        "stride_sec": STRIDE_SEC,
    }
