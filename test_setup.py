"""
test_setup.py
─────────────────────────────────────────────────────────────────────────────
Quick smoke-test: verifies the full pipeline works end-to-end using
SYNTHETIC data (no CICIDS2017 CSV required).
Run this first to make sure all dependencies are installed correctly.

Usage:
    python test_setup.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import numpy as np

print("="*60)
print("  IoT Attack Detection — Setup Verification Test")
print("="*60)

# ── Dependency check ───────────────────────────────────────────
print("\n[1/6] Checking imports …")
try:
    import tensorflow as tf
    import sklearn
    import pandas as pd
    import matplotlib
    import seaborn
    import joblib
    import tqdm
    import imblearn
    print(f"  ✓ TensorFlow  : {tf.__version__}")
    print(f"  ✓ scikit-learn: {sklearn.__version__}")
    print(f"  ✓ pandas      : {pd.__version__}")
    print(f"  ✓ imbalanced-learn installed")
except ImportError as e:
    print(f"  ✗ Missing dependency: {e}")
    print("  → Run: pip install -r requirements.txt")
    sys.exit(1)

# ── Synthetic dataset ──────────────────────────────────────────
print("\n[2/6] Generating synthetic traffic data …")
np.random.seed(42)
N_SAMPLES  = 2000
N_FEATURES = 30

X_synth = np.random.rand(N_SAMPLES, N_FEATURES).astype(np.float32)
y_synth = (np.random.rand(N_SAMPLES) > 0.7).astype(int)   # 30% attacks
print(f"  ✓ Shape: {X_synth.shape}  |  Attack rate: {y_synth.mean()*100:.0f}%")

# ── Sliding window ─────────────────────────────────────────────
print("\n[3/6] Testing sliding window …")
from utils.sliding_window import create_sliding_windows
WINDOW = 10
X_w, y_w = create_sliding_windows(X_synth, y_synth, window_size=WINDOW, step_size=5)
print(f"  ✓ Windows created: {X_w.shape}")

# ── Build model ───────────────────────────────────────────────
print("\n[4/6] Building LSTM model …")
from models.lstm_model import build_lstm_model
input_shape = (WINDOW, N_FEATURES)
model = build_lstm_model(
    input_shape=input_shape,
    lstm_units_1=32, lstm_units_2=16,
    dense_units=8, dropout_rate=0.2,
    learning_rate=0.001
)
model.summary()
print("  ✓ Model built successfully")

# ── Quick train ───────────────────────────────────────────────
print("\n[5/6] Running mini training (5 epochs) …")
split = int(0.8 * len(X_w))
X_tr, X_v = X_w[:split], X_w[split:]
y_tr, y_v = y_w[:split], y_w[split:]

history = model.fit(
    X_tr, y_tr, validation_data=(X_v, y_v),
    epochs=5, batch_size=64, verbose=1
)
print("  ✓ Training complete")

# ── Evaluate ──────────────────────────────────────────────────
print("\n[6/6] Testing metrics …")
from models.lstm_model import predict as model_predict
from utils.metrics import compute_metrics, print_metrics

y_pred, y_prob = model_predict(model, X_v)
metrics = compute_metrics(y_v, y_pred, y_prob)
print_metrics(metrics, title="Smoke-Test Results (Synthetic Data)")

print("\n" + "="*60)
print("  ✅  Setup verification PASSED!")
print("  You can now run train.py with the real CICIDS2017 dataset.")
print("="*60)
