"""
feature_extraction.py
----------------------
Purpose : Convert raw audio files into a labeled MFCC feature dataset.

How it works:
1. Reads every .wav / .mp3 file inside data/real/ and data/fake/
2. Slices each file into 2-second, non-overlapping segments
3. Extracts 20 MFCC coefficients per segment (averaged over time)
4. Saves everything into a single features.csv file

Run this once, before train_model.py.
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa

# ---------------- CONFIG ----------------
SEGMENT_DURATION = 2.0        # seconds per chunk
N_MFCC = 20                   # number of MFCC coefficients
SAMPLE_RATE = 16000            # resample everything to a consistent rate
DATA_DIR = "data"
OUTPUT_CSV = "features.csv"
LABELS = {"real": 0, "fake": 1}   # 0 = Human, 1 = AI Fake
# -----------------------------------------


def slice_audio(y, sr, segment_duration=SEGMENT_DURATION):
    """Split a waveform array into fixed-length segments. Drops the last
    segment if it's shorter than segment_duration (keeps features consistent)."""
    segment_samples = int(segment_duration * sr)
    total_segments = len(y) // segment_samples
    segments = []
    for i in range(total_segments):
        start = i * segment_samples
        end = start + segment_samples
        segments.append(y[start:end])
    return segments


def extract_mfcc(segment, sr, n_mfcc=N_MFCC):
    """Convert one audio segment into a fixed-length MFCC feature vector."""
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)   # average across time -> shape (n_mfcc,)


def process_file(filepath, label):
    """Load one audio file, slice it, extract features for every segment."""
    rows = []
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  [skip] could not read {filepath}: {e}")
        return rows

    segments = slice_audio(y, sr)
    for seg in segments:
        features = extract_mfcc(seg, sr)
        row = list(features) + [label]
        rows.append(row)
    return rows


def build_dataset():
    all_rows = []
    for folder_name, label in LABELS.items():
        folder_path = os.path.join(DATA_DIR, folder_name)
        files = glob.glob(os.path.join(folder_path, "*.wav")) + \
                glob.glob(os.path.join(folder_path, "*.mp3"))

        print(f"Processing '{folder_name}' folder -> {len(files)} files found")
        for filepath in files:
            rows = process_file(filepath, label)
            all_rows.extend(rows)
            print(f"  {os.path.basename(filepath)} -> {len(rows)} segments")

    if not all_rows:
        print("\nNo audio files found. Add files inside data/real/ and data/fake/ first.")
        return

    columns = [f"mfcc_{i+1}" for i in range(N_MFCC)] + ["label"]
    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. {len(df)} total segments saved to {OUTPUT_CSV}")
    print(df["label"].value_counts().rename({0: "real", 1: "fake"}))


if __name__ == "__main__":
    build_dataset()
