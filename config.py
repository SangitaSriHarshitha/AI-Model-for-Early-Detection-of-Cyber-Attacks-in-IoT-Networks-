"""
config.py - Central configuration for IoT Attack Detection Project
Project: AI Model for Early Detection of Cyber Attacks in IoT Networks
Dataset: CICIDS2017
"""

import os

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

for _dir in [MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─────────────────────────────────────────────
#  CICIDS2017 DATASET
# ─────────────────────────────────────────────
# Label column name in CICIDS2017
LABEL_COLUMN = " Label"

# Map multi-class labels → binary (BENIGN=0, ATTACK=1)
BINARY_MODE = True

# Class-imbalance handling. Applied to the training split after scaling and
# before sliding-window creation.
BALANCING_STRATEGY = "smote"  # "smote" | "undersample" | "none"

# Tuned on validation probabilities using a precision-recall curve.
DEFAULT_DECISION_THRESHOLD = 0.5

# Selected feature columns (top features after variance/correlation filtering)
# Set to None to use all features
SELECTED_FEATURES = None

# Train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ─────────────────────────────────────────────
#  SLIDING WINDOW
# ─────────────────────────────────────────────
WINDOW_SIZE = 10      # Number of consecutive traffic records per window
STEP_SIZE   = 1       # Stride between windows (1 = maximum overlap)

# ─────────────────────────────────────────────
#  LSTM MODEL (DEFAULT HYPERPARAMETERS)
# ─────────────────────────────────────────────
LSTM_UNITS_1    = 128
LSTM_UNITS_2    = 64
DENSE_UNITS     = 32
DROPOUT_RATE    = 0.3
LEARNING_RATE   = 0.001
BATCH_SIZE      = 256
EPOCHS          = 50
EARLY_STOP_PAT  = 10   # Patience for early stopping

# ─────────────────────────────────────────────
#  GENETIC ALGORITHM (GA) HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────
GA_POPULATION   = 10
GA_GENERATIONS  = 5
GA_MUTATION_RATE= 0.1
GA_CROSSOVER    = 0.8

# ─────────────────────────────────────────────
#  PARTICLE SWARM OPTIMIZATION (PSO)
# ─────────────────────────────────────────────
PSO_PARTICLES   = 20
PSO_ITERATIONS  = 10
PSO_INERTIA     = 0.5
PSO_COGNITIVE   = 1.5
PSO_SOCIAL      = 1.5

# ─────────────────────────────────────────────
#  HYPERPARAMETER SEARCH SPACE
# ─────────────────────────────────────────────
HYPERPARAM_SPACE = {
    "lstm_units_1":  [64, 128, 256],
    "lstm_units_2":  [32, 64, 128],
    "dense_units":   [16, 32, 64],
    "dropout_rate":  [0.1, 0.2, 0.3, 0.4],
    "learning_rate": [0.0001, 0.001, 0.01],
    "batch_size":    [128, 256, 512],
}

# ─────────────────────────────────────────────
#  OPTIMIZATION MODE
# ─────────────────────────────────────────────
# Options: "none" | "ga" | "pso"
OPTIMIZATION_MODE = "none"

# ─────────────────────────────────────────────
#  MISC
# ─────────────────────────────────────────────
RANDOM_SEED = 42
MAX_SAMPLES = 200_000   # Limit rows for memory safety (set None for full dataset)
