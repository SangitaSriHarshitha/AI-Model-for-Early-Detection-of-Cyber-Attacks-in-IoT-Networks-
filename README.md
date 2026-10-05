# IoT Attack Detection

AI-based early detection of cyber attacks in IoT network traffic using the
CICIDS2017 dataset, sliding-window time-series preprocessing, and a
Bidirectional LSTM with attention.

## What Is Included

- Data loading and preprocessing for CICIDS2017 CSV files
- Sliding-window sequence creation for LSTM input
- Bidirectional LSTM + attention model
- Optional GA/PSO hyperparameter optimization
- Training, evaluation, and CLI prediction scripts
- Local web dashboard with CSV upload prediction
- Saved feature metadata so inference uses the same feature order as training

## Project Structure

```text
iot_attack_detection/
  data/                  CICIDS2017 CSV files
  frontend/              Browser dashboard files
  logs/                  Training and TensorBoard logs
  models/                Saved model, scaler, feature metadata
  optimization/          GA and PSO optimizers
  results/               Reports, plots, predictions
  utils/                 Data, metrics, plotting, metadata helpers
  config.py              Project configuration
  train.py               Training pipeline
  evaluate.py            Test-set evaluation
  predict.py             CLI prediction/demo script
  frontend_server.py     Local dashboard + /predict API server
  requirements.txt       Python dependencies
```

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Place the CICIDS2017 CSV files inside `data/`.

## Train The Model

```bash
python train.py
```

Training writes these important artifacts:

- `models/best_lstm_model.keras`
- `models/scaler.pkl`
- `models/feature_metadata.json`
- `models/best_hyperparams.json`

`feature_metadata.json` is required for reliable prediction because it stores
the exact feature order used during training.

## Evaluate

```bash
python evaluate.py
```

Evaluation writes reports and plots into `results/`.

## CLI Prediction

Run the built-in demo:

```bash
python predict.py
```

Predict from a CSV file:

```bash
python predict.py --input path/to/traffic.csv
```

## Web Dashboard

Start the local web app:

```bash
python frontend_server.py
```

Open:

```text
http://localhost:5000/
```

The dashboard supports:

- CSV upload
- Local `/predict` API call
- Prediction window table
- Total windows, warning count, attack count, and average risk
- Demo simulation when you do not want to upload a CSV

## Notes

- If `models/feature_metadata.json` is missing, prediction falls back to numeric
  column order. Re-run `train.py` to create the metadata file.
- Uploaded CSVs should contain the same traffic feature columns used during
  training. Missing training columns are filled with `0`.
- TensorFlow on native Windows CPU may print informational warnings. These are
  not project errors.
