"""
CLI prediction script for the IoT attack detection model.

Usage:
    python predict.py --input path/to/new_traffic.csv
    python predict.py
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

from config import MODEL_DIR, RESULTS_DIR, WINDOW_SIZE
from models.lstm_model import load_saved_model
from utils.feature_metadata import align_features, load_feature_names
from utils.threshold import load_threshold


EARLY_ALERT_THRESHOLD = 0.35


def preprocess_input(
    df: pd.DataFrame,
    scaler,
    feature_names: list[str] | None,
) -> tuple[np.ndarray, dict]:
    """Align incoming CSV data to the training feature order and scale it."""
    X, alignment = align_features(
        df,
        feature_names=feature_names,
        expected_features=int(scaler.n_features_in_),
    )
    return scaler.transform(X), alignment


def alert_for_probability(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "ATTACK DETECTED"
    if probability >= EARLY_ALERT_THRESHOLD:
        return "EARLY WARNING"
    return "NORMAL"


def stream_predict(
    X_scaled: np.ndarray,
    model,
    threshold: float,
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """Slide a window over scaled traffic and predict each window."""
    rows = []
    for start in range(0, len(X_scaled) - window_size + 1):
        end = start + window_size
        window = X_scaled[start:end][np.newaxis, :, :]
        probability = float(model.predict(window, verbose=0).squeeze())
        rows.append(
            {
                "window_start": start,
                "window_end": end - 1,
                "attack_prob": probability,
                "alert": alert_for_probability(probability, threshold),
            }
        )
    return pd.DataFrame(rows)


def print_alignment_warnings(alignment: dict) -> None:
    if alignment["mode"] == "numeric_fallback":
        print("  Warning: feature metadata not found; using numeric column order.")
        print(
            "  Re-run train.py to create models/feature_metadata.json "
            "for reliable CSV alignment."
        )
    elif alignment["missing_features"]:
        print(
            f"  Warning: {len(alignment['missing_features'])} training feature(s) "
            "were missing from the CSV and filled with 0."
        )


def simulate_demo(model, n_features: int, threshold: float) -> None:
    """Run a dashboard-style synthetic stream demo using scaled features."""
    print("\n[Demo] Simulating IoT network traffic stream (500 packets)...")
    np.random.seed(42)

    normal_traffic = np.random.uniform(0.0, 0.3, (300, n_features))
    attack_traffic = np.random.uniform(0.6, 1.0, (200, n_features))
    attack_traffic[:, : min(5, n_features)] = np.random.uniform(
        0.85, 1.0, (200, min(5, n_features))
    )

    gradual = np.linspace(0, 1, 50)[:, None]
    transition = normal_traffic[-50:] * (1 - gradual) + attack_traffic[:50] * gradual
    stream = np.vstack([normal_traffic[:250], transition, attack_traffic])

    results_df = stream_predict(stream.astype(np.float32), model, threshold)
    print_prediction_summary(results_df)

    out_path = os.path.join(RESULTS_DIR, "demo_predictions.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved -> {out_path}")
    print("\n  Sample predictions:")
    print(results_df.head(15).to_string(index=False))


def predict_from_file(
    csv_path: str,
    model,
    scaler,
    feature_names: list[str] | None,
    threshold: float,
) -> None:
    """Run predictions on a user-provided CSV file."""
    print(f"\n[Predict] Loading file: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    X_scaled, alignment = preprocess_input(df, scaler, feature_names)
    print_alignment_warnings(alignment)
    print(f"  Loaded {len(X_scaled):,} rows | {X_scaled.shape[1]} features")

    if len(X_scaled) < WINDOW_SIZE:
        print(f"  Need at least {WINDOW_SIZE} rows for prediction. Got {len(X_scaled)}.")
        return

    results_df = stream_predict(X_scaled, model, threshold)
    print_prediction_summary(results_df)

    out_path = os.path.join(RESULTS_DIR, "predictions.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n  Predictions saved -> {out_path}")
    print(results_df.to_string(index=False))


def print_prediction_summary(results_df: pd.DataFrame) -> None:
    attacks = results_df[results_df["alert"] == "ATTACK DETECTED"]
    warnings = results_df[results_df["alert"] == "EARLY WARNING"]
    average_risk = results_df["attack_prob"].mean() if len(results_df) else 0.0

    print("\n  Stream Analysis Summary")
    print("  " + "-" * 40)
    print(f"  Total windows analysed : {len(results_df)}")
    print(f"  Early warnings         : {len(warnings)}")
    print(f"  Attack detections      : {len(attacks)}")
    print(f"  Average risk           : {average_risk:.2%}")

    if len(warnings) > 0:
        print(f"  First warning window   : {warnings['window_start'].min()}")
    if len(attacks) > 0:
        print(f"  First attack window    : {attacks['window_start'].min()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IoT Attack Detection prediction script"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a traffic CSV file. If omitted, runs a demo simulation.",
    )
    args = parser.parse_args()

    model_path = os.path.join(MODEL_DIR, "best_lstm_model.keras")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not os.path.exists(model_path):
        print("Model not found. Run train.py first.")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print("Scaler not found. Run train.py first.")
        sys.exit(1)

    print("[Predict] Loading model, scaler, and feature metadata...")
    model = load_saved_model("best_lstm_model.keras")
    scaler = joblib.load(scaler_path)
    feature_names = load_feature_names(required=False)
    threshold = load_threshold()
    print(f"[Predict] Decision threshold: {threshold:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.input:
        predict_from_file(args.input, model, scaler, feature_names, threshold)
    else:
        simulate_demo(model, int(scaler.n_features_in_), threshold)


if __name__ == "__main__":
    main()
