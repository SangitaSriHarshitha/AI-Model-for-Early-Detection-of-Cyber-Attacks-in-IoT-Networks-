"""
evaluate.py
─────────────────────────────────────────────────────────────────────────────
Evaluates the trained LSTM model on the held-out test set.
Generates full report: metrics, confusion matrix, ROC curve.
─────────────────────────────────────────────────────────────────────────────
Usage:
    python evaluate.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import numpy as np
import joblib

from config import (
    MODEL_DIR, RESULTS_DIR, LOGS_DIR,
    TRAIN_RATIO, VAL_RATIO,
    WINDOW_SIZE, STEP_SIZE,
    RANDOM_SEED
)
from utils.data_loader import (
    load_cicids2017, select_features, split_dataset
)
from utils.feature_metadata import load_feature_names
from utils.sliding_window import create_sliding_windows
from models.lstm_model import load_saved_model, predict as model_predict
from utils.metrics import (
    compute_metrics, print_metrics, get_classification_report
)
from utils.threshold import load_threshold
from utils.visualizer import (
    plot_confusion_matrix, plot_roc_curve, plot_metrics_bar
)


def main():
    print("\n" + "═"*60)
    print("  IoT Attack Detection — Test Set Evaluation")
    print("═"*60)

    # ── Load and preprocess (same pipeline as training) ────────
    print("\n[1/4] Loading dataset …")
    df = load_cicids2017()
    df = select_features(df)
    feature_names = load_feature_names(required=False)
    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[feature_names + ["Label"]]

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = split_dataset(
        df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO
    )

    # Load the same scaler fitted during training
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "Scaler not found. Run train.py first."
        )
    scaler = joblib.load(scaler_path)
    X_te_s = scaler.transform(X_te)

    # Sliding windows on test set only
    print("\n[2/4] Applying sliding window to test split …")
    X_test_w, y_test_w = create_sliding_windows(
        X_te_s, y_te, window_size=WINDOW_SIZE, step_size=STEP_SIZE
    )

    # ── Load model ─────────────────────────────────────────────
    print("\n[3/4] Loading saved model …")
    model = load_saved_model("best_lstm_model.keras")

    # ── Evaluate ───────────────────────────────────────────────
    print("\n[4/4] Running inference on test set …")
    threshold = load_threshold()
    y_pred, y_prob = model_predict(model, X_test_w, threshold=threshold)

    is_multiclass = np.asarray(y_prob).ndim == 2 and np.asarray(y_prob).shape[1] > 1
    average = "macro" if is_multiclass else "binary"
    metrics = compute_metrics(y_test_w, y_pred, y_prob, average=average)
    if not is_multiclass:
        metrics["decision_threshold"] = threshold
    print_metrics(metrics, title="TEST SET METRICS")

    # Classification report
    target_names = (
        [f"Class {i}" for i in sorted(np.unique(y_test_w).astype(int))]
        if is_multiclass
        else ["BENIGN", "ATTACK"]
    )
    report = get_classification_report(
        y_test_w, y_pred, target_names=target_names
    )
    print("\nDetailed Classification Report:\n")
    print(report)

    # Save report
    report_path = os.path.join(RESULTS_DIR, "test_report.txt")
    with open(report_path, "w") as f:
        f.write("IoT Attack Detection — Test Set Report\n")
        f.write("="*50 + "\n\n")
        f.write("Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k:<20}: {v:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
    print(f"\nReport saved → {report_path}")

    # Save metrics JSON
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    # ── Plots ──────────────────────────────────────────────────
    plot_confusion_matrix(y_test_w, y_pred,
                           class_names=target_names)
    if not is_multiclass:
        plot_roc_curve(y_test_w, y_prob)
    plot_metrics_bar(metrics)

    print("\n" + "═"*60)
    print("  Evaluation Complete!")
    print("═"*60)


if __name__ == "__main__":
    main()
