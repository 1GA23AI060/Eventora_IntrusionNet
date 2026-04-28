"""Train/export the model from the user's notebook workflow.

The notebook at D:\\newtork.ipynb trains successfully in-memory, but it does not
save the CNN-LSTM model, scaler, encoders, or feature order. This script keeps
the same overall workflow and writes backend-compatible artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "model" / "artifacts"
DEFAULT_TRAIN = ROOT / "data" / "UNSW_NB15_training-set.csv"
DEFAULT_TEST = ROOT / "data" / "UNSW_NB15_testing-set.csv"


def build_model(input_dim: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim, 1)),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export backend artifacts from the notebook training workflow.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN), help="UNSW-NB15 training CSV path.")
    parser.add_argument("--test", default=str(DEFAULT_TEST), help="UNSW-NB15 testing CSV path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    test_path = Path(args.test)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "UNSW-NB15 CSV files are missing. Put them in data/ or pass --train and --test paths."
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    df.columns = [str(column).strip() for column in df.columns]

    encoders = {}
    for column in ["proto", "service", "state"]:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str).str.strip())
        encoders[column] = encoder

    drop_columns = [column for column in ["label", "attack_cat", "id"] if column in df.columns]
    x_frame = df.drop(columns=drop_columns)
    y = df["label"].astype(int)

    scaler = StandardScaler()
    x = scaler.fit_transform(x_frame).astype("float32")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model(x_train.shape[1])
    model.fit(
        x_train.reshape(x_train.shape[0], x_train.shape[1], 1),
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        verbose=1,
    )

    probabilities = model.predict(x_test.reshape(x_test.shape[0], x_test.shape[1], 1), verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    model.save(ARTIFACT_DIR / "model.h5")
    preprocessor_bundle = {
        "scaler": scaler,
        "label_encoders": encoders,
        "feature_columns": list(x_frame.columns),
        "kind": "notebook_labelencoder_scaler",
    }
    joblib.dump(preprocessor_bundle, ARTIFACT_DIR / "preprocessor.joblib")
    joblib.dump(preprocessor_bundle, ARTIFACT_DIR / "notebook_preprocessor.joblib")
    metadata = {
        "feature_columns": list(x_frame.columns),
        "target_column": "label",
        "model_type": "notebook_cnn_lstm",
        "threshold": 0.5,
        "input_dim": int(x_train.shape[1]),
        "preprocessor_type": "notebook_labelencoder_scaler",
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model artifacts to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
