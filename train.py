"""
train.py
─────────────────────────────────────────────────────────────────────────────
Main training pipeline for:
  "AI Model for Early Detection of Cyber Attacks in IoT Networks"
  Dataset : CICIDS2017
  Model   : Bidirectional LSTM + Attention
  Optimizer: Genetic Algorithm (GA) or Particle Swarm Optimization (PSO)
─────────────────────────────────────────────────────────────────────────────
Usage:
    python train.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import numpy as np
import joblib

# ── Project imports ────────────────────────────────────────────
from config import (
    MODEL_DIR, RESULTS_DIR, LOGS_DIR,
    TRAIN_RATIO, VAL_RATIO,
    WINDOW_SIZE, STEP_SIZE,
    LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS,
    DROPOUT_RATE, LEARNING_RATE,
    BATCH_SIZE, EPOCHS, EARLY_STOP_PAT,
    OPTIMIZATION_MODE, RANDOM_SEED,
    BALANCING_STRATEGY, BINARY_MODE
)
from utils.data_loader import (
    load_cicids2017, select_features, scale_features,
    balance_dataset, split_dataset
)
from utils.feature_metadata import save_feature_metadata
from utils.sliding_window import create_windows_for_splits
from models.lstm_model import build_lstm_model, train_model
from utils.metrics import compute_metrics, print_metrics
from utils.threshold import save_threshold, tune_threshold_by_f1
from utils.visualizer import (
    plot_training_history, plot_confusion_matrix,
    plot_roc_curve, plot_metrics_bar, plot_optimization_history
)

np.random.seed(RANDOM_SEED)


def main():
    print("\n" + "═"*60)
    print("  IoT Cyber Attack Early Detection — Training Pipeline")
    print("═"*60)

    # ── 1. Load & clean dataset ────────────────────────────────
    print("\n[Step 1/7] Loading CICIDS2017 dataset …")
    df = load_cicids2017()

    # ── 2. Feature selection ───────────────────────────────────
    print("\n[Step 2/7] Feature selection …")
    df = select_features(df)
    feature_names = [c for c in df.columns if c != "Label"]
    print(f"  Features used: {len(feature_names)}")
    metadata_path = save_feature_metadata(feature_names)
    print(f"  Feature metadata saved -> {metadata_path}")

    # ── 3. Temporal train/val/test split ───────────────────────
    print("\n[Step 3/7] Splitting dataset …")
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = split_dataset(
        df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO
    )

    # ── 4. Scale features ──────────────────────────────────────
    print("\n[Step 4/7] Scaling features (MinMax) …")
    X_tr, X_v, X_te, _ = scale_features(X_tr, X_v, X_te)

    if BALANCING_STRATEGY != "none":
        print(f"\n[Step 4b/7] Balancing training split with {BALANCING_STRATEGY}...")
        X_tr, y_tr = balance_dataset(X_tr, y_tr, strategy=BALANCING_STRATEGY)

    # ── 5. Sliding window (time-series windows) ────────────────
    print("\n[Step 5/7] Applying sliding window …")
    splits = create_windows_for_splits(
        X_tr, y_tr, X_v, y_v, X_te, y_te,
        window_size=WINDOW_SIZE, step_size=STEP_SIZE
    )
    X_train_w = splits["X_train"];  y_train_w = splits["y_train"]
    X_val_w   = splits["X_val"];    y_val_w   = splits["y_val"]
    X_test_w  = splits["X_test"];   y_test_w  = splits["y_test"]

    input_shape = (WINDOW_SIZE, X_train_w.shape[2])
    label_count = len(np.unique(y_train_w))
    model_classes = 1 if BINARY_MODE and label_count <= 2 else label_count
    print(f"  Input shape for LSTM: {input_shape}")
    print(f"  Model output classes: {model_classes}")

    # ── 6. Hyperparameter optimization ────────────────────────
    best_params = {
        "lstm_units_1": LSTM_UNITS_1,
        "lstm_units_2": LSTM_UNITS_2,
        "dense_units":  DENSE_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "learning_rate":LEARNING_RATE,
        "batch_size":   BATCH_SIZE,
    }
    opt_history = []

    if OPTIMIZATION_MODE == "ga":
        print("\n[Step 6/7] Running Genetic Algorithm optimization …")
        from optimization.genetic_algorithm import run_genetic_algorithm
        best_params, best_fit, opt_history = run_genetic_algorithm(
            X_train_w, y_train_w, X_val_w, y_val_w, input_shape,
            n_classes=model_classes
        )
        plot_optimization_history(opt_history, method="GA")

    elif OPTIMIZATION_MODE == "pso":
        print("\n[Step 6/7] Running PSO optimization …")
        from optimization.pso_optimizer import run_pso
        best_params, best_fit, opt_history = run_pso(
            X_train_w, y_train_w, X_val_w, y_val_w, input_shape,
            n_classes=model_classes
        )
        plot_optimization_history(opt_history, method="PSO")

    else:
        print("\n[Step 6/7] Skipping optimization (using default hyperparams) …")

    # Save best params
    params_path = os.path.join(MODEL_DIR, "best_hyperparams.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Best hyperparams saved → {params_path}")

    # ── 7. Final model training ────────────────────────────────
    print("\n[Step 7/7] Training final LSTM model …")
    model = build_lstm_model(
        input_shape   = input_shape,
        n_classes     = model_classes,
        lstm_units_1  = best_params["lstm_units_1"],
        lstm_units_2  = best_params["lstm_units_2"],
        dense_units   = best_params["dense_units"],
        dropout_rate  = best_params["dropout_rate"],
        learning_rate = best_params["learning_rate"],
    )

    history = train_model(
        model,
        X_train_w, y_train_w,
        X_val_w,   y_val_w,
        batch_size = best_params["batch_size"],
        epochs     = EPOCHS,
        patience   = EARLY_STOP_PAT,
        model_name = "best_lstm_model.keras"
    )

    # ── Plots ──────────────────────────────────────────────────
    print("\n[Results] Generating plots …")
    plot_training_history(history)

    # Validation set evaluation
    from models.lstm_model import predict as model_predict
    y_pred_val, y_prob_val = model_predict(model, X_val_w)

    threshold = None
    if model_classes == 1:
        threshold, threshold_metrics = tune_threshold_by_f1(y_val_w, y_prob_val)
        threshold_path = save_threshold(threshold, threshold_metrics)
        print(f"[Threshold] Tuned validation threshold: {threshold:.4f}")
        print(f"[Threshold] Saved -> {threshold_path}")
        print(f"[Threshold] decision_threshold.json contents: threshold={threshold:.4f}")
        y_pred_val, y_prob_val = model_predict(model, X_val_w, threshold=threshold)

    average = "binary" if model_classes == 1 else "macro"
    val_metrics = compute_metrics(y_val_w, y_pred_val, y_prob_val, average=average)
    if threshold is not None:
        val_metrics["decision_threshold"] = threshold
    print_metrics(val_metrics, title="Validation Metrics")

    class_names = ["BENIGN", "ATTACK"] if model_classes == 1 else [
        f"Class {i}" for i in range(model_classes)
    ]
    plot_confusion_matrix(y_val_w, y_pred_val,
                           class_names=class_names)
    if model_classes == 1:
        plot_roc_curve(y_val_w, y_prob_val)
    plot_metrics_bar(val_metrics)

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "val_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in val_metrics.items()}, f, indent=2)

    print("\n" + "═"*60)
    print("  Training Complete!")
    print(f"  Model saved  → {os.path.join(MODEL_DIR, 'best_lstm_model.keras')}")
    print(f"  Results dir  → {RESULTS_DIR}")
    print("═"*60)


if __name__ == "__main__":
    main()
