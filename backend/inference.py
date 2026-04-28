"""Model loading and prediction utilities for the Flask API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import sys

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from preprocess import align_input_columns, clean_features, to_cnn_lstm_shape  # noqa: E402


class IntrusionDetector:
    """Loads saved artifacts once and exposes row/CSV prediction methods."""

    def __init__(
        self,
        model_path: Path,
        preprocessor_path: Path,
        metadata_path: Path,
        demo_model_path: Path | None = None,
        demo_metadata_path: Path | None = None,
    ) -> None:
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.metadata_path = metadata_path
        self.demo_model_path = demo_model_path
        self.demo_metadata_path = demo_metadata_path
        self.model: Any | None = None
        self.preprocessor: Any | None = None
        self.metadata: dict[str, Any] | None = None
        self.loaded_mode: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._real_ready or self._demo_ready

    @property
    def mode(self) -> str:
        if self._real_ready:
            return "production"
        if self._demo_ready:
            return "demo"
        return "missing"

    @property
    def _real_ready(self) -> bool:
        return self.model_path.exists() and self.preprocessor_path.exists() and self.metadata_path.exists()

    @property
    def _demo_ready(self) -> bool:
        return bool(
            self.demo_model_path
            and self.demo_metadata_path
            and self.demo_model_path.exists()
            and self.demo_metadata_path.exists()
        )

    def load(self) -> None:
        if self._real_ready:
            import tensorflow as tf

            self.model = tf.keras.models.load_model(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.loaded_mode = "production"
            return

        if self._demo_ready and self.demo_model_path and self.demo_metadata_path:
            self.model = joblib.load(self.demo_model_path)
            self.preprocessor = None
            self.metadata = json.loads(self.demo_metadata_path.read_text(encoding="utf-8"))
            self.loaded_mode = "demo"
            return

        if not self.is_ready:
            missing = [
                str(path)
                for path in [self.model_path, self.preprocessor_path, self.metadata_path]
                if not path.exists()
            ]
            raise FileNotFoundError(
                "Model artifacts are missing. Train the model first. Missing: " + ", ".join(missing)
            )

    def predict_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if self.model is None or self.preprocessor is None or self.metadata is None:
            self.load()

        feature_columns = self.metadata["feature_columns"]
        threshold = float(self.metadata.get("threshold", 0.5))

        frame = align_input_columns(rows, feature_columns)
        if self.loaded_mode == "demo":
            probabilities = self.model.predict_proba(frame)[:, 1]
        else:
            transformed = self._transform_production_frame(frame)
            probabilities = self.model.predict(to_cnn_lstm_shape(transformed), verbose=0).ravel()
        labels = (probabilities >= threshold).astype(int)

        results = [
            {
                "row": int(index),
                "prediction": "Attack" if int(label) == 1 else "Normal",
                "attack_probability": round(float(probability), 6),
                "severity": self._severity(float(probability)),
            }
            for index, (label, probability) in enumerate(zip(labels, probabilities), start=1)
        ]

        attack_count = int(np.sum(labels == 1))
        normal_count = int(np.sum(labels == 0))
        attack_rate = 0.0 if len(results) == 0 else attack_count / len(results)
        average_attack_probability = 0.0 if len(results) == 0 else float(np.mean(probabilities))
        return {
            "results": results,
            "summary": {
                "total": len(results),
                "attack": attack_count,
                "normal": normal_count,
                "attack_rate": round(float(attack_rate), 6),
                "average_attack_probability": round(average_attack_probability, 6),
                "high_confidence_attacks": int(np.sum(probabilities >= 0.85)),
                "threshold": threshold,
                "model_mode": self.loaded_mode or self.mode,
            },
        }

    def predict_csv(self, csv_file: Any) -> dict[str, Any]:
        frame = pd.read_csv(csv_file)
        frame.columns = [str(col).strip() for col in frame.columns]
        for column in ["label", "attack_cat", "id"]:
            if column in frame.columns:
                frame = frame.drop(columns=[column])
        frame = clean_features(frame)
        return self.predict_rows(frame.to_dict(orient="records"))

    @staticmethod
    def _severity(probability: float) -> str:
        if probability >= 0.90:
            return "Critical"
        if probability >= 0.75:
            return "High"
        if probability >= 0.50:
            return "Elevated"
        return "Low"

    def _transform_production_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if isinstance(self.preprocessor, dict) and {"scaler", "label_encoders", "feature_columns"}.issubset(
            self.preprocessor.keys()
        ):
            transformed = frame.copy()
            for column, encoder in self.preprocessor["label_encoders"].items():
                values = transformed[column].astype(str).str.strip()
                known_classes = set(encoder.classes_)
                fallback = encoder.classes_[0]
                values = values.where(values.isin(known_classes), fallback)
                transformed[column] = encoder.transform(values)
            transformed = transformed[self.preprocessor["feature_columns"]]
            return self.preprocessor["scaler"].transform(transformed).astype("float32")

        if isinstance(self.preprocessor, dict):
            raise ValueError(f"Unsupported preprocessor bundle: {self.preprocessor.get('kind', 'unknown')}")

        return self.preprocessor.transform(frame).astype("float32")
