"""Train ANN and CNN-LSTM intrusion detection models on UNSW-NB15."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocess import build_feature_pipeline, load_dataset, to_cnn_lstm_shape


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "UNSW_NB15_training-set.csv"
DEFAULT_TEST = ROOT / "data" / "UNSW_NB15_testing-set.csv"
ARTIFACT_DIR = ROOT / "model" / "artifacts"


def build_ann(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="ann_baseline",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def build_cnn_lstm(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(input_dim, 1))
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.SpatialDropout1D(0.20)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True))(x)
    attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = tf.keras.layers.Add()([x, attention])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_bilstm_attention_intrusion_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def tune_threshold(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    min_precision: float,
) -> tuple[float, dict]:
    """Choose a decision threshold that favors precision without ignoring recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    candidates = []
    for index, threshold in enumerate(thresholds):
        p = precision[index]
        r = recall[index]
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        candidates.append((float(threshold), float(p), float(r), float(f1)))

    eligible = [item for item in candidates if item[1] >= min_precision]
    selected = max(eligible or candidates, key=lambda item: item[3])
    return selected[0], {
        "selected_threshold": selected[0],
        "validation_precision": selected[1],
        "validation_recall": selected[2],
        "validation_f1": selected[3],
        "target_min_precision": min_precision,
        "target_met": bool(eligible),
    }


def evaluate_model(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray, threshold: float) -> dict:
    probabilities = model.predict(x_test, verbose=0).ravel()
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Attack"],
            output_dict=True,
            zero_division=0,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNSW-NB15 ANN and CNN-LSTM models.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN), help="Path to training CSV.")
    parser.add_argument("--test", default=str(DEFAULT_TEST), help="Path to test CSV.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size.")
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.90,
        help="Preferred minimum attack precision when tuning the final threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_data = load_dataset(args.train)
    test_data = load_dataset(args.test)

    preprocessor = build_feature_pipeline(train_data.x)
    x_train = preprocessor.fit_transform(train_data.x).astype("float32")
    x_test = preprocessor.transform(test_data.x).astype("float32")
    y_train = train_data.y.astype(int)
    y_test = test_data.y.astype(int)
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )
    class_labels = np.array([0, 1])
    class_weights = compute_class_weight(class_weight="balanced", classes=class_labels, y=y_fit)
    class_weight = {int(label): float(weight) for label, weight in zip(class_labels, class_weights)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        ),
    ]

    ann = build_ann(x_train.shape[1])
    ann.fit(
        x_fit,
        y_fit,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    cnn_lstm = build_cnn_lstm(x_train.shape[1])
    cnn_lstm.fit(
        to_cnn_lstm_shape(x_fit),
        y_fit,
        validation_data=(to_cnn_lstm_shape(x_val), y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    ann_val_probabilities = ann.predict(x_val, verbose=0).ravel()
    ann_threshold, ann_threshold_report = tune_threshold(ann_val_probabilities, y_val, args.min_precision)
    cnn_val_probabilities = cnn_lstm.predict(to_cnn_lstm_shape(x_val), verbose=0).ravel()
    cnn_threshold, cnn_threshold_report = tune_threshold(cnn_val_probabilities, y_val, args.min_precision)

    ann_metrics = evaluate_model(ann, x_test, y_test, ann_threshold)
    cnn_lstm_metrics = evaluate_model(cnn_lstm, to_cnn_lstm_shape(x_test), y_test, cnn_threshold)
    ann_metrics["threshold_tuning"] = ann_threshold_report
    cnn_lstm_metrics["threshold_tuning"] = cnn_threshold_report

    ann.save(ARTIFACT_DIR / "ann_baseline.h5")
    cnn_lstm.save(ARTIFACT_DIR / "cnn_lstm_model.h5")
    cnn_lstm.save(ARTIFACT_DIR / "model.h5")
    joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.joblib")

    metadata = {
        "feature_columns": list(train_data.x.columns),
        "target_column": "label",
        "model_type": "cnn_bilstm_attention",
        "threshold": float(cnn_threshold),
        "input_dim": int(x_train.shape[1]),
        "class_weight": class_weight,
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics = {
        "ann_baseline": ann_metrics,
        "cnn_lstm": cnn_lstm_metrics,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"ANN accuracy: {ann_metrics['accuracy']:.4f}, precision: {ann_metrics['precision']:.4f}")
    print(
        "CNN-BiLSTM accuracy: "
        f"{cnn_lstm_metrics['accuracy']:.4f}, precision: {cnn_lstm_metrics['precision']:.4f}, "
        f"threshold: {cnn_threshold:.4f}"
    )
    print(f"Artifacts saved to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
