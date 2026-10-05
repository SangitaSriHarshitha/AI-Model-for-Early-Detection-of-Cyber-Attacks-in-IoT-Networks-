# IoT Attack Detection Project - Task Completion Plan

✅ **Training pipeline fixed and completed** (default hyperparameters, no optimization)

## Remaining Steps:

### 2. [DONE] Fix model evaluation
- `models/lstm_model.py` uses a registered custom `AttentionLayer` instead of Lambda layers.
- `load_saved_model()` passes the custom layer and uses safe-mode compatibility for older saved models.

### 3. [DONE] Address class imbalance and weak precision/F1
- SMOTE is applied to the training split after scaling and before sliding-window creation.
- A validation precision-recall curve is used to save `models/decision_threshold.json`.
- Evaluation, CLI prediction, and the dashboard load the saved threshold instead of defaulting to 0.5.

### 4. [PENDING] Test full pipeline
```bash
rm models/best_lstm_model.keras models/scaler.pkl
python train.py
python evaluate.py
ls models/ results/
```

### 5. [PENDING] Verify results
- Check accuracy, F1 >90%
- Plots: training_history.png, confusion_matrix.png, roc_curve.png

### 6. [PENDING] Baseline comparison
```bash
python baseline_models.py
```

### 7. [PENDING] Multi-class experiment
- Set `BINARY_MODE = False` in `config.py`, retrain, and report per-class metrics.

**Next**: Implement fixes per approved plan.
