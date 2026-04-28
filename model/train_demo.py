"""Create a small demo predictor when the real UNSW-NB15 dataset is unavailable.

This is only for UI/API testing. Use train.py with UNSW-NB15 for the real model.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "demo.csv"
ARTIFACT_DIR = ROOT / "model" / "artifacts"


def build_demo_data() -> pd.DataFrame:
    rows = [
        {"dur": 0.05, "proto": "tcp", "service": "http", "state": "FIN", "sbytes": 320, "dbytes": 980, "label": 0},
        {"dur": 0.08, "proto": "tcp", "service": "dns", "state": "CON", "sbytes": 120, "dbytes": 240, "label": 0},
        {"dur": 0.20, "proto": "udp", "service": "dns", "state": "CON", "sbytes": 90, "dbytes": 180, "label": 0},
        {"dur": 1.20, "proto": "tcp", "service": "ftp", "state": "FIN", "sbytes": 1200, "dbytes": 4200, "label": 0},
        {"dur": 6.80, "proto": "tcp", "service": "-", "state": "SYN", "sbytes": 60, "dbytes": 0, "label": 1},
        {"dur": 8.10, "proto": "tcp", "service": "ssh", "state": "SYN", "sbytes": 48, "dbytes": 0, "label": 1},
        {"dur": 3.50, "proto": "udp", "service": "-", "state": "INT", "sbytes": 1500, "dbytes": 20, "label": 1},
        {"dur": 10.5, "proto": "tcp", "service": "http", "state": "RST", "sbytes": 5200, "dbytes": 10, "label": 1},
        {"dur": 0.12, "proto": "tcp", "service": "http", "state": "FIN", "sbytes": 420, "dbytes": 1100, "label": 0},
        {"dur": 9.40, "proto": "icmp", "service": "-", "state": "INT", "sbytes": 4000, "dbytes": 0, "label": 1},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame = build_demo_data()
    frame.to_csv(DATA_PATH, index=False)

    feature_columns = ["dur", "proto", "service", "state", "sbytes", "dbytes"]
    x = frame[feature_columns]
    y = frame["label"]

    numeric_features = ["dur", "sbytes", "dbytes"]
    categorical_features = ["proto", "service", "state"]
    pipeline = Pipeline(
        steps=[
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        ("numeric", StandardScaler(), numeric_features),
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                    ]
                ),
            ),
            ("classifier", RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")),
        ]
    )
    pipeline.fit(x, y)

    joblib.dump(pipeline, ARTIFACT_DIR / "demo_model.joblib")
    metadata = {
        "feature_columns": feature_columns,
        "threshold": 0.5,
        "model_type": "demo_random_forest",
        "warning": "Demo model for UI/API testing only. Train train.py with UNSW-NB15 for real results.",
    }
    (ARTIFACT_DIR / "demo_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Demo model saved to {ARTIFACT_DIR / 'demo_model.joblib'}")
    print(f"Demo CSV saved to {DATA_PATH}")


if __name__ == "__main__":
    main()
