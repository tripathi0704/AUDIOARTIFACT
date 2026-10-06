"""
train_full_pipeline.py
----------------------
End-to-end automated pipeline to:
  1. Extract 768-dim WavLM embeddings from ALL 2,000+ audio files in data/real & data/fake
  2. Train and calibrate the XGBoost Classifier
  3. Validate on user's test.wav and test2.wav audio files
  4. Save the production model to model/deepfake_detector.pkl
"""

import os
import sys
import time

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from feature_extraction import build_dataset
from train_model import train
from backend.main import analyze_audio_data

FEATURES_CSV = "features_wavlm.csv"
MODEL_PATH = os.path.join("model", "deepfake_detector.pkl")

def run():
    print("=" * 70)
    print("   STARTING AUDIOARTIFACT FULL 2,000-FILE PRODUCTION TRAINING PIPELINE   ")
    print("=" * 70)
    overall_start = time.time()

    # Step 1: Full Feature Extraction (max_files_per_class=0 means ALL files)
    print("\n>>> STAGE 1: Full WavLM Feature Extraction across all audio files...")
    build_dataset(max_files_per_class=0, output_csv=FEATURES_CSV, batch_size=32)

    # Step 2: Model Training
    print("\n>>> STAGE 2: Training XGBoost Classifier on full dataset...")
    train(features_csv=FEATURES_CSV)

    # Step 3: Automated Validation on user's personal test files
    print("\n>>> STAGE 3: Running Post-Training Validation on User Test Clips...")
    test_files = [
        r"C:\Users\shubh\Downloads\test.wav",
        r"C:\Users\shubh\Downloads\test2.wav",
    ]

    for tf in test_files:
        if os.path.exists(tf):
            print(f"\nEvaluating: {tf}")
            with open(tf, "rb") as f:
                content = f.read()
            res = analyze_audio_data(content, original_filename=os.path.basename(tf))
            print(f"  Duration    : {res.get('total_duration')}s")
            print(f"  Segments    : {res.get('segments_count')}")
            print(f"  Fake Ratio  : {res.get('fake_ratio')}%")
            print(f"  Verdict     : {res.get('verdict')}")
        else:
            print(f"  File not found for validation: {tf}")

    elapsed_mins = (time.time() - overall_start) / 60
    print("\n" + "=" * 70)
    print(f"   FULL PIPELINE COMPLETE! Total Time: {elapsed_mins:.1f} minutes")
    print(f"   New Model Saved to: {MODEL_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    run()
