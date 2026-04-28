"""Flask API for AI-based network intrusion detection."""

from __future__ import annotations

import json
import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import BadRequest

from config import (
    DEMO_METADATA_PATH,
    DEMO_MODEL_PATH,
    FRONTEND_DIR,
    MAX_UPLOAD_SIZE_MB,
    METADATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
)
from inference import IntrusionDetector


app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024
CORS(app)

detector = IntrusionDetector(
    MODEL_PATH,
    PREPROCESSOR_PATH,
    METADATA_PATH,
    demo_model_path=DEMO_MODEL_PATH,
    demo_metadata_path=DEMO_METADATA_PATH,
)


@app.get("/")
def index():
    """Serve the frontend if present, otherwise return a health response."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return send_from_directory(FRONTEND_DIR, "index.html")
    return jsonify({"status": "ok", "message": "Intrusion Detection API is running"})


@app.get("/health")
def health():
    threshold = None
    missing_artifacts = []
    if METADATA_PATH.exists():
        threshold = json.loads(METADATA_PATH.read_text(encoding="utf-8")).get("threshold")
    for path in [MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH]:
        if not path.exists():
            missing_artifacts.append(path.name)
    if threshold is None and DEMO_METADATA_PATH.exists():
        threshold = json.loads(DEMO_METADATA_PATH.read_text(encoding="utf-8")).get("threshold")

    return jsonify(
        {
            "status": "ok",
            "model_ready": detector.is_ready,
            "model_mode": detector.mode,
            "threshold": threshold,
            "missing_artifacts": missing_artifacts,
            "message": "Train the model first if model_ready is false.",
        }
    )


@app.post("/predict")
def predict():
    """Predict a single JSON record, a JSON record list, or an uploaded CSV file."""
    try:
        if "file" in request.files:
            uploaded_file = request.files["file"]
            if not uploaded_file.filename:
                raise BadRequest("Uploaded file has no filename.")
            response = detector.predict_csv(uploaded_file)
            return jsonify(response)

        payload = request.get_json(silent=True)
        if not payload:
            raise BadRequest("Send JSON features or upload a CSV file under the 'file' field.")

        if "features" in payload:
            rows = payload["features"]
        else:
            rows = payload

        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise BadRequest("'features' must be an object or an array of objects.")

        response = detector.predict_rows(rows)
        return jsonify(response)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except BadRequest as exc:
        return jsonify({"error": exc.description}), 400
    except Exception as exc:
        return jsonify({"error": "Prediction failed.", "details": str(exc)}), 500


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": f"Uploaded file is too large. Limit is {MAX_UPLOAD_SIZE_MB} MB."}), 413


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug)
