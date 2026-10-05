"""
Train classical baseline models for comparison with BiLSTM + Attention.

This script answers the examiner question "why LSTM?" by training three
tabular baselines on the same cleaned/scaled feature set:
  - Random Forest
  - MLPClassifier
  - XGBoost

It writes:
  results/baseline_comparison.csv
  results/baseline_comparison.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from config import (
    BALANCING_STRATEGY,
    BINARY_MODE,
    MODEL_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
)
from utils.data_loader import (
    balance_dataset,
    load_cicids2017,
    scale_features,
    select_features,
    split_dataset,
)
from utils.feature_metadata import save_feature_metadata
from utils.metrics import compute_metrics
from utils.threshold import tune_threshold_by_f1


def predict_with_optional_threshold(model, X_val, y_val, X_test):
    y_prob_val = model.predict_proba(X_val)
    y_prob_test = model.predict_proba(X_test)

    if BINARY_MODE and y_prob_test.shape[1] == 2:
        threshold, threshold_metrics = tune_threshold_by_f1(y_val, y_prob_val[:, 1])
        y_pred_test = (y_prob_test[:, 1] >= threshold).astype(int)
        return y_pred_test, y_prob_test[:, 1], threshold, threshold_metrics, "binary"

    y_pred_test = np.argmax(y_prob_test, axis=1)
    return y_pred_test, y_prob_test, None, {}, "macro"


def train_and_score(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"[Baseline] Training {name}...")
    model.fit(X_train, y_train)
    y_pred, y_prob, threshold, threshold_metrics, average = predict_with_optional_threshold(
        model,
        X_val,
        y_val,
        X_test,
    )
    metrics = compute_metrics(y_test, y_pred, y_prob, average=average)
    metrics["model"] = name
    if threshold is not None:
        metrics["decision_threshold"] = threshold
        metrics["validation_f1_at_threshold"] = threshold_metrics.get("f1_score", 0.0)
    return metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = select_features(load_cicids2017())
    feature_names = [c for c in df.columns if c != "Label"]
    save_feature_metadata(feature_names)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(
        df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "baseline_scaler.pkl"))

    if BALANCING_STRATEGY != "none":
        X_train, y_train = balance_dataset(
            X_train,
            y_train,
            strategy=BALANCING_STRATEGY,
        )

    models = [
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
        ),
        (
            "MLP",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=80,
                early_stopping=True,
                random_state=RANDOM_SEED,
            ),
        ),
    ]

    if XGBClassifier is None:
        print("[Baseline] XGBoost not installed; install requirements to enable it.")
    else:
        xgb_params = {
            "n_estimators": 200,
            "use_label_encoder": False,
            "eval_metric": "logloss" if BINARY_MODE else "mlogloss",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        if BINARY_MODE:
            xgb_params["scale_pos_weight"] = 65

        models.append(
            (
                "XGBoost",
                XGBClassifier(**xgb_params),
            )
        )

    rows = [
        train_and_score(name, model, X_train, y_train, X_val, y_val, X_test, y_test)
        for name, model in models
    ]

    df_results = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "baseline_comparison.csv")
    json_path = os.path.join(RESULTS_DIR, "baseline_comparison.json")
    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"[Baseline] Saved -> {csv_path}")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
