from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import EXCLUDE_FROM_TRAINING, INFERENCE_NULL_COLUMNS, LABEL_COLUMN


def training_matrix(full_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build X, y and the ordered feature column list used for training."""
    if LABEL_COLUMN not in full_df.columns:
        raise ValueError(f"Expected label column {LABEL_COLUMN!r} in full_df")

    y = full_df[LABEL_COLUMN].fillna(0).astype(int)
    feature_cols = [c for c in full_df.columns if c not in EXCLUDE_FROM_TRAINING and c != LABEL_COLUMN]
    X = full_df[feature_cols].copy()
    return X, y, feature_cols


def prepare_inference_features(
    full_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Align columns to training schema; null out position (unknown at rank time)."""
    X = full_df.copy()
    for col in INFERENCE_NULL_COLUMNS:
        if col in X.columns:
            X[col] = pd.NA
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(f"Inference frame missing training features: {missing}")
    return X[feature_columns]


def save_feature_columns(path: str | Path, feature_columns: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")


def load_feature_columns(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
