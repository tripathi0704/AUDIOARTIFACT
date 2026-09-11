"""
train_model.py
---------------
Purpose : Train an XGBoost classifier on the 768-dimensional Microsoft WavLM
          embeddings (features_wavlm.csv) extracted via VAD and 50% overlapping windows.
          Evaluates Accuracy, Precision/Recall/F1, Confusion Matrix, and ROC-AUC,
          then saves the model for continuous probability predictions in the backend.

Run this after feature_extraction.py has produced features_wavlm.csv.
"""

import os
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

DEFAULT_FEATURES_CSV = "features_wavlm.csv"
MODEL_DIR = "model"
MODEL_OUTPUT = os.path.join(MODEL_DIR, "deepfake_detector.pkl")


def train(features_csv: str = DEFAULT_FEATURES_CSV):
    if not os.path.exists(features_csv):
        print(f"Error: {features_csv} not found. Please run feature_extraction.py first.")
        return

    print(f"Loading dataset from {features_csv}...")
    df = pd.read_csv(features_csv)
    X = df.drop(columns=["label"])
    y = df["label"]

    num_features = X.shape[1]
    print(f"Dataset shape: {df.shape} ({num_features} feature columns, {len(df)} segments)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} segments, testing on {len(X_test)} segments")
    y_train_series = pd.Series(y_train)
    print(f"Class balance in training set:\n{y_train_series.value_counts().rename({0: 'real', 1: 'fake'})}")

    # Calculate class balance
    num_fake = int((y_train_series == 1).sum())
    num_real = int((y_train_series == 0).sum())
    scale_pos_weight = (num_real / max(num_fake, 1))
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.3f}")

    print("\nTraining XGBoost Classifier on 768-dim WavLM embeddings...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Predictions & Probabilities
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probas)

    print(f"\n================ MODEL EVALUATION ================")
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Test ROC-AUC:  {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Human (Real)", "AI Fake"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("===================================================\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    print(f"Model successfully saved to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Classifier on WavLM embeddings")
    parser.add_argument("--features", type=str, default=DEFAULT_FEATURES_CSV, help="Path to features CSV")
    args = parser.parse_args()
    train(features_csv=args.features)
