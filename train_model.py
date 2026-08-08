"""
train_model.py
---------------
Purpose : Train a classifier on features.csv (created by feature_extraction.py)
          and save the trained model for use in the backend.

Run this after feature_extraction.py has produced features.csv.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

FEATURES_CSV = "features.csv"
MODEL_OUTPUT = "model/deepfake_detector.pkl"


def main():
    df = pd.read_csv(FEATURES_CSV)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} segments, testing on {len(X_test)} segments")
    print(f"Class balance in training set:\n{y_train.value_counts().rename({0:'real', 1:'fake'})}")

    # ---- handle class imbalance ----
    # if fake segments greatly outnumber real segments, the model learns to
    # always predict "fake" since that alone already gives high accuracy.
    # scale_pos_weight (XGBoost standard formula = negative/positive count)
    # down-weights the dominant "fake" class so real segments matter more.
    num_fake = (y_train == 1).sum()   # positive class
    num_real = (y_train == 0).sum()   # negative class
    scale_pos_weight = num_real / num_fake

    print(f"scale_pos_weight set to: {scale_pos_weight:.3f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["real", "fake"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    joblib.dump(model, MODEL_OUTPUT)
    print(f"\nModel saved to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
