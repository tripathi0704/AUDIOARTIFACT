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
import types
import tempfile
from contextlib import asynccontextmanager

# Windows 11 Smart App Control & cross-platform safe numba bypass for librosa
try:
    from numba import _dispatcher
except Exception:
    _m = types.ModuleType("numba")
    _m.jit = lambda *a, **kw: (lambda f: f) if a and callable(a[0]) else (lambda f: f)
    _m.njit = _m.jit
    _m.vectorize = _m.jit
    _m.guvectorize = _m.jit
    _m.stencil = _m.jit
    _m.prange = range
    sys.modules["numba"] = _m
    sys.modules["numba.core"] = types.ModuleType("numba.core")
    sys.modules["numba.core.decorators"] = _m

import joblib
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# allow importing local modules
sys.path.append(os.path.dirname(__file__))
from db import init_db, save_result, get_history  # noqa: E402
from vad_utils import filter_speech_vad  # noqa: E402
from forensic_utils import (  # noqa: E402
    compute_file_hashes,
    compute_acoustic_forensics,
    compute_speaker_similarity,
    generate_forensic_html_report,
    slice_segment_audio_bytes,
)
from ai_copilot import (  # noqa: E402
    generate_forensic_explanation,
    assess_scam_threat,
    chat_with_audio_copilot,
    is_genai_installed,
)


# ---------------- CONFIG ----------------
WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"
WINDOW_SEC = 2.0              # 2-second analysis window
STRIDE_SEC = 1.0              # 1-second stride -> 50% overlap
SAMPLE_RATE = 16000
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "deepfake_detector.pkl"))
LABEL_NAMES = {0: "Human", 1: "AI Fake"}
# -----------------------------------------

model = None
_wavlm_extractor = None
_wavlm_model = None


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


_wavlm_error = None


def get_wavlm():
    """Lazily load and cache WavLM feature extractor and model on CPU."""
    global _wavlm_extractor, _wavlm_model, _wavlm_error
    if _wavlm_model is None:
        try:
            print(f"[Backend] Loading WavLM model ({WAVLM_MODEL_ID})...")
            try:
                # Fast path: load from local cache if available (localhost)
                _wavlm_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_ID, local_files_only=True)
                _wavlm_model = AutoModel.from_pretrained(WAVLM_MODEL_ID, low_cpu_mem_usage=True, local_files_only=True)
            except Exception as e_local:
                # Cloud fallback: download and cache from HuggingFace (Streamlit Community Cloud)
                print(f"[Backend] Not in local cache ({e_local}). Downloading {WAVLM_MODEL_ID} from HuggingFace...")
                _wavlm_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_ID)
                _wavlm_model = AutoModel.from_pretrained(WAVLM_MODEL_ID, low_cpu_mem_usage=True)
            _wavlm_model.eval()
            print("[Backend] WavLM model successfully loaded.")
            _wavlm_error = None
        except Exception as e:
            import traceback
            _wavlm_error = f"{type(e).__name__}: {e}"
            print(f"[Backend] Error loading WavLM: {_wavlm_error}")
            traceback.print_exc()
    return _wavlm_extractor, _wavlm_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    get_wavlm()
    yield


app = FastAPI(title="AudioArtifact API v2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



def slice_overlapping(y: np.ndarray, sr: int = SAMPLE_RATE, window_sec: float = WINDOW_SEC, stride_sec: float = STRIDE_SEC):
    """
    Slice audio into 50% overlapping windows: 2.0s window with 1.0s stride.
    Guarantees complete coverage of the full audio timeline from 0.0s to total_duration.
    """
    win_len = int(window_sec * sr)
    stride_len = int(stride_sec * sr)
    total_len = len(y)
    total_sec = round(total_len / sr, 2)
    segments = []

    if total_len <= win_len:
        padded = np.pad(y, (0, max(0, win_len - total_len)), mode="constant")
        segments.append((0.0, max(total_sec, window_sec), padded))
        return segments

    last_end = 0
    for start in range(0, total_len - win_len + 1, stride_len):
        end = start + win_len
        segments.append((round(start / sr, 2), round(end / sr, 2), y[start:end]))
        last_end = end

    # If remaining tail audio >= 0.2s, add a final window reaching the exact end of file
    if last_end < total_len and (total_len - last_end) >= int(0.2 * sr):
        tail_start = max(0, total_len - win_len)
        tail_wave = y[tail_start:total_len]
        if len(tail_wave) < win_len:
            tail_wave = np.pad(tail_wave, (0, win_len - len(tail_wave)), mode="constant")
        segments.append((round(tail_start / sr, 2), total_sec, tail_wave))

    return segments


def decode_audio_bytes(content: bytes, original_filename: str = "upload.wav"):
    """
    Decodes audio bytes to 16kHz mono float32 array.
    Uses in-memory soundfile decoding for WAV/OGG/FLAC, falls back to tempfile + librosa for MP3/M4A.
    """
    y = None
    # 1. Fast in-memory audio decoding
    try:
        import io
        import soundfile as sf
        with io.BytesIO(content) as bio:
            audio_data, sr_orig = sf.read(bio, dtype="float32")
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            if sr_orig != SAMPLE_RATE:
                y = librosa.resample(audio_data, orig_sr=sr_orig, target_sr=SAMPLE_RATE)
            else:
                y = audio_data
    except Exception:
        y = None

    # Fallback to tempfile + librosa for compressed formats like MP3/M4A
    if y is None:
        suffix = os.path.splitext(original_filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            return None, SAMPLE_RATE, f"Failed to decode audio file: {e}"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if y is None or len(y) == 0:
        return None, SAMPLE_RATE, "Failed to decode audio file. Please ensure it is a valid .wav or .mp3 file."

    return y, SAMPLE_RATE, None


def analyze_audio_data(content: bytes, original_filename: str = "upload.wav") -> dict:
    """
    High-performance forensic analysis engine with in-memory audio decoding
    and vectorized batch inference across the entire continuous audio timeline.
    """
    clf = get_model()
    if clf is None:
        return {"error": "Detection model not loaded. Please train the model using train_model.py first."}

    # 1. Decode audio bytes
    y, sr_int, err = decode_audio_bytes(content, original_filename)
    if err:
        return {"error": err}

    raw_duration = round(len(y) / SAMPLE_RATE, 2)
    file_hashes = compute_file_hashes(content)
    forensic_signals = compute_acoustic_forensics(y, sr_int)

    # Normalize peak amplitude to standard 0.95 so soft microphone recordings are evaluated at standard gain
    max_amp = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
    if max_amp > 0.001:
        y_norm = (y / max_amp) * 0.95
    else:
        y_norm = y

    # 2. Voice Activity Detection (silence tracking & acoustic diagnostics)
    y_speech, speech_intervals = filter_speech_vad(y_norm, sr_int)
    forensic_signals["speech_intervals"] = speech_intervals
    forensic_signals["active_speech_duration_sec"] = round(len(y_speech) / sr_int, 2)

    # 3. Continuous Temporal Sliding Window across the FULL audio timeline
    # Slices the entire recording from 0.0s to raw_duration so NO seconds are lost!
    segments = slice_overlapping(y_norm, sr_int, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC)

    if not segments:
        return {
            "error": f"Audio clip is too short ({raw_duration}s). Minimum recommended length is 2 seconds."
        }

    # 4. High-speed vectorized batch WavLM prediction (768-dim embeddings)
    extractor, wavlm_model = get_wavlm()
    if extractor is None or wavlm_model is None:
        err_detail = _wavlm_error if _wavlm_error else "Model weights could not be loaded."
        return {"error": f"WavLM model failed to initialize: {err_detail}"}

    seg_waves = [seg for _, _, seg in segments]
    with torch.inference_mode():
        if len(seg_waves) <= 16:
            inputs = extractor(seg_waves, sampling_rate=sr_int, return_tensors="pt", padding=True)
            out = wavlm_model(**inputs)
            feats = out.last_hidden_state.mean(dim=1).cpu().numpy()
        else:
            feat_chunks = []
            for b_idx in range(0, len(seg_waves), 16):
                chunk = seg_waves[b_idx:b_idx + 16]
                inputs = extractor(chunk, sampling_rate=sr_int, return_tensors="pt", padding=True)
                out = wavlm_model(**inputs)
                feat_chunks.append(out.last_hidden_state.mean(dim=1).cpu().numpy())
            feats = np.vstack(feat_chunks)

    probas = clf.predict_proba(feats)  # shape: (N, 2)
    fake_probs = probas[:, 1]
    human_probs = probas[:, 0]
    preds = (fake_probs >= 0.50).astype(int)

    results = []
    fake_count = int(np.sum(preds))
    highest_idx = int(np.argmax(fake_probs))
    highest_fake_prob = float(fake_probs[highest_idx])
    highest_conf = round(float((highest_fake_prob if preds[highest_idx] == 1 else human_probs[highest_idx]) * 100), 2)

    highest_risk = {
        "segment": highest_idx,
        "start": segments[highest_idx][0],
        "end": segments[highest_idx][1],
        "fake_probability": round(highest_fake_prob, 4),
        "confidence": highest_conf,
    }

    for i, (start_t, end_t, _) in enumerate(segments):
        p_fake = float(fake_probs[i])
        p_human = float(human_probs[i])
        pr = int(preds[i])
        conf = round((p_fake if pr == 1 else p_human) * 100, 2)

        results.append({
            "segment": i,
            "start": start_t,
            "end": end_t,
            "label": LABEL_NAMES[pr],
            "label_code": pr,
            "fake_probability": round(p_fake, 4),
            "confidence": conf,
        })

    # Summary forensic calculations
    total_analyzed_duration = round(segments[-1][1], 2)
    fake_ratio = round((fake_count / len(segments)) * 100, 1)
    fake_seconds = round(fake_count * STRIDE_SEC, 2)

    if fake_ratio <= 15.0:
        verdict = "Authentic Human Voice"
    elif fake_ratio >= 75.0:
        verdict = "Fully Synthetic Voice Detected"
    else:
        verdict = "Spliced Audio Detected"

    # 4. Mel-Spectrogram voice signature payload for visual reporting
    try:
        mel_spec = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=32)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        max_frames = 100
        if mel_db.shape[1] > max_frames:
            idx = np.linspace(0, mel_db.shape[1] - 1, max_frames).astype(int)
            mel_db = mel_db[:, idx]

        time_axis = np.linspace(0, raw_duration, mel_db.shape[1]).round(2).tolist()
        mel_axis = list(range(mel_db.shape[0]))
        spec_dict = {
            "z": mel_db.round(2).tolist(),
            "time": time_axis,
            "mel": mel_axis,
        }
    except Exception:
        spec_dict = {}

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
        "spectrogram": spec_dict,
        "file_hashes": file_hashes,
        "forensic_signals": forensic_signals,
    }

    # 5. Persist run to SQLite history
    try:
        save_result(original_filename, response)
    except Exception as e:
        print(f"[Backend] Warning: could not save to history.db: {e}")

    return response


def extract_speaker_embedding_vector(content: bytes, original_filename: str = "upload.wav"):
    """
    Extracts mean-pooled 768-dimensional WavLM speaker representation from speech segments.
    Used for cross-audio voiceprint matching and deepfake clone verification.
    """
    y, sr_int, err = decode_audio_bytes(content, original_filename)
    if err:
        return None, err
    if y is None or len(y) == 0:
        return None, "Empty audio buffer."

    # Normalize and apply VAD with fallback
    max_amp = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
    y_norm = (y / max_amp * 0.95) if max_amp > 0.001 else y

    y_speech, _ = filter_speech_vad(y_norm, sr_int)
    if len(y_speech) < int(SAMPLE_RATE * 1.5):
        y_speech = y_norm

    segments = slice_overlapping(y_speech, sr_int, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC)
    if not segments:
        return None, "Audio is too short for speaker voiceprint extraction (minimum 2 seconds required)."

    extractor, wavlm_model = get_wavlm()
    if extractor is None or wavlm_model is None:
        return None, "WavLM model not loaded."

    seg_waves = [seg for _, _, seg in segments]
    with torch.inference_mode():
        if len(seg_waves) <= 16:
            inputs = extractor(seg_waves, sampling_rate=sr_int, return_tensors="pt", padding=True)
            out = wavlm_model(**inputs)
            feats = out.last_hidden_state.mean(dim=1).cpu().numpy()
        else:
            feat_chunks = []
            for b_idx in range(0, len(seg_waves), 16):
                chunk = seg_waves[b_idx:b_idx + 16]
                inputs = extractor(chunk, sampling_rate=sr_int, return_tensors="pt", padding=True)
                out = wavlm_model(**inputs)
                feat_chunks.append(out.last_hidden_state.mean(dim=1).cpu().numpy())
            feats = np.vstack(feat_chunks)

    mean_emb = np.mean(feats, axis=0)
    return mean_emb, None


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    return analyze_audio_data(content, file.filename or "upload.wav")


@app.post("/compare_speakers")
async def compare_speakers(file_ref: UploadFile = File(...), file_suspect: UploadFile = File(...)):
    """
    Compares two speaker recordings via WavLM 768-dim embeddings.
    Returns cosine similarity, distance, and clone matching assessment.
    """
    content_ref = await file_ref.read()
    content_suspect = await file_suspect.read()

    emb_ref, err_ref = extract_speaker_embedding_vector(content_ref, file_ref.filename or "reference.wav")
    if err_ref:
        return {"error": f"Reference audio error: {err_ref}"}

    emb_suspect, err_suspect = extract_speaker_embedding_vector(content_suspect, file_suspect.filename or "suspect.wav")
    if err_suspect:
        return {"error": f"Suspect audio error: {err_suspect}"}

    sim_result = compute_speaker_similarity(emb_ref, emb_suspect)
    sim_result["reference_filename"] = file_ref.filename or "reference.wav"
    sim_result["suspect_filename"] = file_suspect.filename or "suspect.wav"
    return sim_result


@app.post("/export_report", response_class=HTMLResponse)
async def export_report(payload: dict):
    """Generates and returns a standalone, printable HTML forensic audit certificate."""
    html = generate_forensic_html_report(payload)
    return HTMLResponse(content=html, status_code=200)


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
        "genai_installed": is_genai_installed(),
    }


@app.post("/ai/explain")
async def ai_explain(
    file: Optional[UploadFile] = File(None),
    analysis: str = Form(...),
    language: str = Form("English"),
    api_key: Optional[str] = Form(None),
):
    try:
        analysis_dict = json.loads(analysis) if isinstance(analysis, str) else analysis
    except Exception:
        analysis_dict = {}
    audio_bytes = await file.read() if file else None
    fname = file.filename if file else analysis_dict.get("filename", "audio.wav")
    return generate_forensic_explanation(audio_bytes, fname, analysis_dict, language=language, api_key=api_key)


@app.post("/ai/threat_assessment")
async def ai_threat(
    file: Optional[UploadFile] = File(None),
    analysis: str = Form(...),
    api_key: Optional[str] = Form(None),
):
    try:
        analysis_dict = json.loads(analysis) if isinstance(analysis, str) else analysis
    except Exception:
        analysis_dict = {}
    audio_bytes = await file.read() if file else None
    fname = file.filename if file else analysis_dict.get("filename", "audio.wav")
    return assess_scam_threat(audio_bytes, fname, analysis_dict, api_key=api_key)


@app.post("/ai/chat")
async def ai_chat(
    file: Optional[UploadFile] = File(None),
    query: str = Form(...),
    analysis: str = Form(...),
    chat_history: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
):
    try:
        analysis_dict = json.loads(analysis) if isinstance(analysis, str) else analysis
    except Exception:
        analysis_dict = {}
    try:
        history_list = json.loads(chat_history) if chat_history else None
    except Exception:
        history_list = None
    audio_bytes = await file.read() if file else None
    fname = file.filename if file else analysis_dict.get("filename", "audio.wav")
    return chat_with_audio_copilot(audio_bytes, fname, query, analysis_dict, chat_history=history_list, api_key=api_key)


