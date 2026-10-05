"""
Local frontend + prediction API server.

Routes:
  GET  /          -> dashboard
  GET  /predict   -> dashboard, for browser refreshes on that URL
  POST /predict   -> multipart CSV upload prediction endpoint
"""

import json
import os
from email import policy
from email.parser import BytesParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

from config import MODEL_DIR, WINDOW_SIZE
from utils.feature_metadata import align_features, load_feature_names
from utils.threshold import load_threshold


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_NAME = "best_lstm_model.keras"
SCALER_NAME = "scaler.pkl"
ARTIFACT_CACHE = {}
PREDICT_BATCH_SIZE = 2048
MAX_RETURNED_WINDOWS = 1000
WARNING_THRESHOLD = 0.35


class FrontendHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("", "/", "/predict"):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            self.send_error(404, "Not found")
            return

        try:
            payload = self._handle_prediction_upload()
            self._send_json(payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _handle_prediction_upload(self):
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        file_bytes, file_name = extract_uploaded_file(body, content_type)
        df = pd.read_csv(file_bytes, low_memory=False)
        return predict_dataframe(df, source_name=file_name)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def load_artifacts():
    if ARTIFACT_CACHE:
        return (
            ARTIFACT_CACHE["model"],
            ARTIFACT_CACHE["scaler"],
            ARTIFACT_CACHE["feature_names"],
            ARTIFACT_CACHE["threshold"],
        )

    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    scaler_path = os.path.join(MODEL_DIR, SCALER_NAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run train.py first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found. Run train.py first.")

    from models.lstm_model import load_saved_model

    ARTIFACT_CACHE["model"] = load_saved_model(MODEL_NAME)
    ARTIFACT_CACHE["scaler"] = joblib.load(scaler_path)
    ARTIFACT_CACHE["feature_names"] = load_feature_names(required=False)
    ARTIFACT_CACHE["threshold"] = load_threshold()
    return (
        ARTIFACT_CACHE["model"],
        ARTIFACT_CACHE["scaler"],
        ARTIFACT_CACHE["feature_names"],
        ARTIFACT_CACHE["threshold"],
    )


def predict_dataframe(df: pd.DataFrame, source_name: str | None = None) -> dict:
    model, scaler, feature_names, threshold = load_artifacts()
    expected_features = int(scaler.n_features_in_)

    X, alignment = align_features(df, feature_names, expected_features)
    X_scaled = scaler.transform(X)

    if len(X_scaled) < WINDOW_SIZE:
        raise ValueError(f"Need at least {WINDOW_SIZE} rows for prediction.")

    probabilities = predict_probabilities_batched(model, X_scaled, WINDOW_SIZE)
    rows, display_start = prediction_rows(
        probabilities,
        WINDOW_SIZE,
        MAX_RETURNED_WINDOWS,
        threshold,
    )
    summary = prediction_summary(
        probabilities,
        returned_windows=len(rows),
        threshold=threshold,
        display_start=display_start,
    )

    source = {
        "file_name": source_name,
        "rows_received": int(len(df)),
        "columns_received": int(len(df.columns)),
    }

    return {
        "predictions": rows,
        "summary": summary,
        "alignment": alignment,
        "source": source,
    }


def predict_probabilities_batched(model, X_scaled: np.ndarray, window_size: int) -> np.ndarray:
    """Run model predictions in batches instead of one TensorFlow call per window."""
    n_windows = len(X_scaled) - window_size + 1
    probabilities = np.empty(n_windows, dtype=np.float32)

    for batch_start in range(0, n_windows, PREDICT_BATCH_SIZE):
        batch_end = min(batch_start + PREDICT_BATCH_SIZE, n_windows)
        batch_len = batch_end - batch_start
        batch = np.empty(
            (batch_len, window_size, X_scaled.shape[1]),
            dtype=np.float32,
        )

        for offset, window_start in enumerate(range(batch_start, batch_end)):
            batch[offset] = X_scaled[window_start : window_start + window_size]

        probabilities[batch_start:batch_end] = model.predict(
            batch,
            verbose=0,
        ).reshape(-1)

    return probabilities


def prediction_rows(
    probabilities: np.ndarray,
    window_size: int,
    max_rows: int,
    threshold: float,
) -> tuple[list[dict], int]:
    display_start = choose_display_start(probabilities, max_rows)
    display_end = min(display_start + max_rows, len(probabilities))

    rows = []
    for start in range(display_start, display_end):
        probability = probabilities[start]
        prob = float(probability)
        rows.append(
            {
                "window_start": start,
                "window_end": start + window_size - 1,
                "attack_prob": prob,
                "alert": alert_for_probability(prob, threshold),
            }
        )
    return rows, display_start


def choose_display_start(probabilities: np.ndarray, max_rows: int) -> int:
    if len(probabilities) <= max_rows:
        return 0

    first_page = probabilities[:max_rows]
    if np.any(first_page >= WARNING_THRESHOLD):
        return 0

    risky_mask = probabilities >= WARNING_THRESHOLD
    if not np.any(risky_mask):
        return 0

    first_risky = int(np.argmax(risky_mask))
    return max(0, min(first_risky - 50, len(probabilities) - max_rows))


def prediction_summary(
    probabilities: np.ndarray,
    returned_windows: int,
    threshold: float,
    display_start: int,
) -> dict:
    attack_mask = probabilities >= threshold
    warning_mask = (probabilities >= WARNING_THRESHOLD) & ~attack_mask
    total_windows = int(len(probabilities))
    display_end = display_start + returned_windows - 1 if returned_windows else None

    return {
        "total_windows": total_windows,
        "returned_windows": int(returned_windows),
        "truncated": returned_windows < total_windows,
        "display_start": int(display_start),
        "display_end": display_end,
        "warning_count": int(warning_mask.sum()),
        "attack_count": int(attack_mask.sum()),
        "average_risk": float(probabilities.mean()) if total_windows else 0.0,
        "max_risk": float(probabilities.max()) if total_windows else 0.0,
        "decision_threshold": float(threshold),
    }


def extract_uploaded_file(body: bytes, content_type: str):
    if "multipart/form-data" not in content_type:
        if body:
            return BytesIO(body), None
        raise ValueError("CSV upload body is empty.")

    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\n"
        b"MIME-Version: 1.0\r\n\r\n" + body
    )

    for part in message.iter_parts():
        disposition = part.get_content_disposition()
        field_name = part.get_param("name", header="content-disposition")
        if disposition == "form-data" and field_name == "file":
            payload = part.get_payload(decode=True)
            if not payload:
                raise ValueError("Uploaded CSV file is empty.")
            filename = part.get_filename()
            return BytesIO(payload), filename

    raise ValueError("CSV file field named 'file' is required.")


def alert_for_probability(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "ATTACK DETECTED"
    if probability >= WARNING_THRESHOLD:
        return "EARLY WARNING"
    return "NORMAL"


def main():
    port = int(os.environ.get("PORT", "5000"))
    server = ThreadingHTTPServer(("localhost", port), FrontendHandler)
    print(f"Dashboard running at http://localhost:{port}")
    print(f"Prediction endpoint: http://localhost:{port}/predict")
    server.serve_forever()


if __name__ == "__main__":
    main()
