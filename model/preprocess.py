"""Preprocessing helpers for UNSW-NB15 intrusion detection models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.feature_selection import VarianceThreshold


TARGET_COLUMN = "label"
DROP_COLUMNS = {"id", "attack_cat"}


@dataclass(frozen=True)
class DatasetBundle:
    x: pd.DataFrame
    y: np.ndarray


def load_dataset(path: str) -> DatasetBundle:
    """Load and clean a UNSW-NB15 CSV file."""
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {path}")

    y = df[TARGET_COLUMN].astype(int).to_numpy()
    drop_cols = [col for col in DROP_COLUMNS | {TARGET_COLUMN} if col in df.columns]
    x = df.drop(columns=drop_cols)
    x = clean_features(x)
    return DatasetBundle(x=x, y=y)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize missing values and infinities without changing column names."""
    cleaned = df.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].astype(str).str.strip()

    return cleaned


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer for numeric and categorical UNSW-NB15 fields."""
    categorical_features = list(x.select_dtypes(include=["object", "category", "bool"]).columns)
    numeric_features = [col for col in x.columns if col not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline(x: pd.DataFrame) -> Pipeline:
    """Create the full preprocessing pipeline used by training and inference."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x)),
            ("variance_filter", VarianceThreshold(threshold=0.0)),
        ]
    )


def align_input_columns(rows: Iterable[dict], feature_columns: list[str]) -> pd.DataFrame:
    """Build a DataFrame from API rows using the exact training feature order."""
    df = pd.DataFrame(list(rows))
    for column in feature_columns:
        if column not in df.columns:
            df[column] = np.nan
    return clean_features(df[feature_columns])


def to_cnn_lstm_shape(x: np.ndarray) -> np.ndarray:
    """Represent tabular features as a sequence for Conv1D + LSTM."""
    return x.reshape((x.shape[0], x.shape[1], 1))
