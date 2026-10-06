"""
feature_extraction.py
----------------------
Purpose : Convert raw audio files into a 768-feature dataset using Microsoft WavLM
          (microsoft/wavlm-base-plus) with Silero VAD silence filtering and 50%
          overlapping sliding windows (2.0s window, 1.0s stride).
"""

import os
import sys
import glob
import time
import types
import argparse
import numpy as np
import pandas as pd
import torch

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

import librosa
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModel

# Support importing vad_utils whether run from root or elsewhere
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
from vad_utils import filter_speech_vad  # noqa: E402

# ---------------- CONFIG ----------------
WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"
WINDOW_SEC = 2.0              # 2-second analysis window
STRIDE_SEC = 1.0              # 1-second stride -> 50% overlap
SAMPLE_RATE = 16000           # 16kHz standard sampling rate
DATA_DIR = "data"
DEFAULT_OUTPUT_CSV = "features_wavlm.csv"
LABELS = {"real": 0, "fake": 1}  # 0 = Human, 1 = AI Fake
EMBEDDING_DIM = 768
# -----------------------------------------


def slice_overlapping(y: np.ndarray, sr: int = SAMPLE_RATE, window_sec: float = WINDOW_SEC, stride_sec: float = STRIDE_SEC):
    """Slice audio array into overlapping windows."""
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


def load_audio_16k(filepath: str) -> np.ndarray:
    """Robust 16kHz mono audio loader supporting WAV and MP3."""
    try:
        data, sr = sf.read(filepath, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
        return data
    except Exception:
        y, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        return y


def build_dataset(max_files_per_class: int = 50, output_csv: str = DEFAULT_OUTPUT_CSV, batch_size: int = 32):
    """Extracts WavLM embeddings with safe sequential preprocessing and batched inference."""
    print(f"Initializing WavLM ({WAVLM_MODEL_ID})...")
    torch.set_num_threads(14)
    extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_ID)
    model = AutoModel.from_pretrained(WAVLM_MODEL_ID)
    model.eval()

    tasks = []
    for folder_name, label in LABELS.items():
        folder_path = os.path.join(DATA_DIR, folder_name)
        all_found = sorted(
            glob.glob(os.path.join(folder_path, "*.wav")) +
            glob.glob(os.path.join(folder_path, "*.mp3"))
        )
        # Always prioritize user-recorded mic audio and hard negatives
        priority_files = [
            f for f in all_found
            if any(k in os.path.basename(f).lower() for k in ["user", "test", "mic", "original", "human"])
        ]
        other_files = [f for f in all_found if f not in priority_files]
        files = priority_files + other_files

        if max_files_per_class and max_files_per_class > 0:
            files = files[:max_files_per_class]

        print(f"Selected {len(files)} files from '{folder_name}' (label={label}, priority={len(priority_files)})")
        for f in files:
            tasks.append((f, label))

    if not tasks:
        print("No audio files found inside data/real/ or data/fake/.")
        return

    total_files = len(tasks)
    print(f"\nProcessing {total_files} total audio files with multi-core PyTorch (14 threads)...")
    start_time = time.time()
    all_segments_with_labels = []

    for idx, (filepath, label) in enumerate(tasks, 1):
        try:
            y = load_audio_16k(filepath)
            sr_int = int(SAMPLE_RATE)
            
            # Slice audio directly to capture authentic speech onsets, transitions, and pauses
            segments = slice_overlapping(y, sr_int, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC)
            
            # Filter out dead digital silence (RMS < 0.006)
            valid_segs = []
            for _, _, seg in segments:
                rms = float(np.sqrt(np.mean(seg**2))) if len(seg) > 0 else 0.0
                if rms >= 0.006:
                    valid_segs.append(seg)
            
            # Cap segments per file to ensure maximum speaker diversity across files
            if len(valid_segs) > 15:
                valid_segs = valid_segs[:15]

            for seg in valid_segs:
                all_segments_with_labels.append((seg, label))
        except Exception as e:
            print(f"  [skip] {os.path.basename(filepath)}: {e}")

        if idx % 10 == 0 or idx == total_files:
            pct = (idx / total_files) * 100
            print(f"  Audio preprocessing: [{idx}/{total_files}] ({pct:.1f}%) | Segments collected: {len(all_segments_with_labels)}", end="\r")

    print(f"\nPreprocessing complete in {time.time() - start_time:.1f}s. Total segments: {len(all_segments_with_labels)}")

    print(f"\nExtracting 768-dim WavLM embeddings in batches of {batch_size}...")
    t2_start = time.time()
    all_rows = []
    total_segs = len(all_segments_with_labels)

    for i in range(0, total_segs, batch_size):
        chunk = all_segments_with_labels[i:i + batch_size]
        waves = [item[0] for item in chunk]
        labels = [item[1] for item in chunk]

        inputs = extractor(waves, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        for emb, lbl in zip(embs, labels):
            all_rows.append(list(emb) + [lbl])

        done = min(i + batch_size, total_segs)
        rate = done / max(time.time() - t2_start, 0.001)
        eta = (total_segs - done) / max(rate, 0.001)
        print(f"  WavLM inference: [{done}/{total_segs}] ({done/total_segs*100:.1f}%) | {rate:.1f} segs/s | ETA: {eta:.0f}s", end="\r")

    print("\n\nSaving dataset to CSV...")
    columns = [f"wavlm_{i}" for i in range(EMBEDDING_DIM)] + ["label"]
    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(output_csv, index=False)

    total_time = time.time() - start_time
    print(f"Done in {total_time:.1f}s ({total_time/60:.2f} mins)!")
    print(f"Saved {len(df)} segments to {output_csv}.")
    print("Class distribution:")
    print(df["label"].value_counts().rename({0: "real", 1: "fake"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WavLM 768-dim Feature Extractor")
    parser.add_argument("--all", action="store_true", help="Process ALL files in data/")
    parser.add_argument("--limit", type=int, default=150, help="Max files per class (default: 150)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_CSV, help="Output CSV filename")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    args = parser.parse_args()

    max_f = 0 if args.all else args.limit
    build_dataset(max_files_per_class=max_f, output_csv=args.output, batch_size=args.batch_size)
