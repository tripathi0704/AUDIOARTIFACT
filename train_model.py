"""
train_model.py
---------------
Purpose : Train an XGBoost classifier on the 60-feature dataset (features.csv)
          extracted via VAD and 50% overlapping windows.
          Evaluates Accuracy, Precision/Recall/F1, Confusion Matrix, and ROC-AUC,
          then saves the model for continuous probability predictions in the backend.

Run this after feature_extraction.py has produced features.csv.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

FEATURES_CSV = "features.csv"
MODEL_DIR = "model"
MODEL_OUTPUT = os.path.join(MODEL_DIR, "deepfake_detector.pkl")


def main():
    if not os.path.exists(FEATURES_CSV):
        print(f"Error: {FEATURES_CSV} not found. Please run feature_extraction.py first.")
        return

    print(f"Loading dataset from {FEATURES_CSV}...")
    df = pd.read_csv(FEATURES_CSV)
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

    # Handle class imbalance if any
    num_fake = int((y_train_series == 1).sum())
    num_real = int((y_train_series == 0).sum())
    scale_pos_weight = (num_real / max(num_fake, 1))
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.3f}")

    print("\nTraining XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.2,
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
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Human (Real)", "AI Fake"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("===================================================\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    print(f"Model successfully saved to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
