"""
main.py
-------
Purpose : FastAPI backend exposing a single /analyze endpoint.

Flow on every request:
  1. Receive uploaded audio file
  2. Slice into 2-second segments
  3. Extract MFCC features per segment
  4. Run the trained model on each segment
  5. Aggregate into timestamped JSON + save to SQLite
  6. Return the JSON to the frontend

Run with:  uvicorn backend.main:app --reload --port 8000
(run this command from the project root, not from inside backend/)
"""

import os
import sys
import tempfile

import joblib
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# allow importing db.py whether uvicorn is launched from project root or backend/
sys.path.append(os.path.dirname(__file__))
from db import init_db, save_result, get_history  # noqa: E402

# ---------------- CONFIG ----------------
SEGMENT_DURATION = 2.0
N_MFCC = 20
SAMPLE_RATE = 16000
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "deepfake_detector.pkl")
LABEL_NAMES = {0: "Human", 1: "AI Fake"}
# -----------------------------------------

app = FastAPI(title="AudioArtifact API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_resources():
    global model
    init_db()
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: no model found at {MODEL_PATH}. Train one first with train_model.py")


def slice_audio(y, sr, segment_duration=SEGMENT_DURATION):
    segment_samples = int(segment_duration * sr)
    total_segments = len(y) // segment_samples
    segments = []
    for i in range(total_segments):
        start = i * segment_samples
        end = start + segment_samples
        segments.append(y[start:end])
    return segments


def extract_mfcc(segment, sr, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Run train_model.py first."}

    # save the upload to a temp file so librosa can read it
    filename = file.filename or "upload.wav"
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        os.remove(tmp_path)

    segments = slice_audio(y, sr)
    results = []
    fake_seconds = 0

    for i, seg in enumerate(segments):
        features = extract_mfcc(seg, sr).reshape(1, -1)
        pred = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0][pred]

        start_time = i * SEGMENT_DURATION
        end_time = start_time + SEGMENT_DURATION

        if pred == 1:
            fake_seconds += SEGMENT_DURATION

        results.append({
            "segment": i,
            "start": start_time,
            "end": end_time,
            "label": LABEL_NAMES[pred],
            "label_code": pred,
            "confidence": round(float(proba) * 100, 2),
        })

    response = {
        "filename": filename,
        "total_duration": len(segments) * SEGMENT_DURATION,
        "fake_seconds": fake_seconds,
        "verdict": "Spliced Audio Detected" if fake_seconds > 0 else "No AI Voice Detected",
        "segments": results,
    }

    save_result(filename, response)
    return response


@app.get("/history")
def history(limit: int = 20):
    return get_history(limit)


@app.get("/")
def root():
    return {"status": "AudioArtifact backend running", "model_loaded": model is not None}
