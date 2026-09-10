"""
feature_extraction.py
----------------------
Purpose : Convert raw audio files into a 60-feature dataset (MFCC + Delta + Delta-Delta)
          with Silero VAD silence filtering and 50% overlapping windows (2s window, 1s stride).

Roadmap v2.0 Pipeline:
1. Load .wav / .mp3 from data/real/ and data/fake/
2. Run Voice Activity Detection (Silero VAD) to filter out non-speech silence
3. Slice speech with 50% overlapping windows (2.0s window, 1.0s stride)
4. Extract 60-feature vector per segment:
     - 20 MFCCs (mean across time)
     - 20 Delta MFCCs (velocity)
     - 20 Delta-Delta MFCCs (acceleration)
5. Save aggregated dataset to features.csv
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import librosa

# Support importing vad_utils whether run from root or elsewhere
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
from vad_utils import filter_speech_vad  # noqa: E402

# ---------------- CONFIG ----------------
WINDOW_SEC = 2.0              # 2-second analysis window
STRIDE_SEC = 1.0              # 1-second stride -> 50% overlap
N_MFCC = 20                   # 20 base MFCC coefficients
SAMPLE_RATE = 16000           # 16kHz standard sampling rate
DATA_DIR = "data"
OUTPUT_CSV = "features.csv"
LABELS = {"real": 0, "fake": 1}  # 0 = Human, 1 = AI Fake
# -----------------------------------------


def extract_advanced_features(y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extracts a 60-dimensional feature vector per audio segment:
    - 20 MFCC coefficients (mean over time)
    - 20 Delta MFCC coefficients (1st order derivative)
    - 20 Delta-Delta MFCC coefficients (2nd order derivative)
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    return np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1),
        np.mean(delta2_mfcc, axis=1)
    ])


def slice_overlapping(y: np.ndarray, sr: int = SAMPLE_RATE, window_sec: float = WINDOW_SEC, stride_sec: float = STRIDE_SEC):
    """
    Slice audio array into overlapping windows.
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


def process_file(filepath: str, label: int, max_segments: int = 40):
    """
    Load one audio file, apply VAD silence removal, slice with 50% overlap,
    and extract 60 features per segment.
    """
    rows = []
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  [skip] could not read {filepath}: {e}")
        return rows

    # 1. Voice Activity Detection (remove silence chunks)
    y_speech, _ = filter_speech_vad(y, sr)
    if len(y_speech) < int(SAMPLE_RATE * 1.0):
        y_speech = y

    # 2. Overlapping slicing (2s window, 1s stride)
    segments = slice_overlapping(y_speech, sr, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC)

    if max_segments and len(segments) > max_segments:
        segments = segments[:max_segments]

    # 3. 60-feature extraction per segment
    for _, _, seg in segments:
        features = extract_advanced_features(seg, sr)
        rows.append(list(features) + [label])

    return rows


def build_dataset(max_files_per_class: int = 150):
    """
    Extracts features across files in data/real and data/fake
    and outputs features.csv.
    """
    all_rows = []
    for folder_name, label in LABELS.items():
        folder_path = os.path.join(DATA_DIR, folder_name)
        files = sorted(
            glob.glob(os.path.join(folder_path, "*.wav")) +
            glob.glob(os.path.join(folder_path, "*.mp3"))
        )

        if max_files_per_class and max_files_per_class > 0:
            files = files[:max_files_per_class]

        print(f"\nProcessing '{folder_name}' folder -> {len(files)} files to extract")
        for i, filepath in enumerate(files):
            rows = process_file(filepath, label)
            all_rows.extend(rows)
            if (i + 1) % 25 == 0 or (i + 1) == len(files):
                print(f"  [{i+1}/{len(files)}] {os.path.basename(filepath)} -> {len(rows)} segments (accumulated: {len(all_rows)})")

    if not all_rows:
        print("\nNo audio files found. Add files inside data/real/ and data/fake/ first.")
        return

    # 60 features: 20 MFCC + 20 Delta + 20 Delta2
    columns = (
        [f"mfcc_{i+1}" for i in range(N_MFCC)] +
        [f"delta_{i+1}" for i in range(N_MFCC)] +
        [f"delta2_{i+1}" for i in range(N_MFCC)] +
        ["label"]
    )
    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataset build complete: {len(df)} total segments saved to {OUTPUT_CSV} ({len(columns)-1} features per row).")
    print("Class distribution:")
    print(df["label"].value_counts().rename({0: "real", 1: "fake"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 60 MFCC+Delta features with VAD and 50% overlap")
    parser.add_argument("--all", action="store_true", help="Process all files in data/ (may take longer)")
    parser.add_argument("--limit", type=int, default=150, help="Max files per class (default: 150)")
    args = parser.parse_args()

    max_files = None if args.all else args.limit
    build_dataset(max_files_per_class=max_files)
