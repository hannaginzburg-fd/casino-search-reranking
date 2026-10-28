from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data.config import DEFAULT_FEATURE_LIST_PATH, DEFAULT_MODEL_PATH, LABEL_COLUMN
from features import save_feature_columns, training_matrix
from load_data import load_train_df_table, save_train_df

REPO_ROOT = Path(__file__).resolve().parents[1]


def train(
    full_df: pd.DataFrame,
    *,
    model_path: Path,
    feature_list_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    X, y, feature_cols = training_matrix(full_df)
    train_df = X.copy()
    train_df[LABEL_COLUMN] = y
    save_train_df(train_df)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    categorical = [
        c
        for c in X.columns
        if X[c].dtype == "object" or str(X[c].dtype) == "boolean" or pd.api.types.is_bool_dtype(X[c])
    ]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=categorical or "auto",
        callbacks=[lgb.early_stopping(50, verbose=True)],
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    metrics = {"val_auc": float(roc_auc_score(y_val, val_proba)), "train_rows": len(X_train), "val_rows": len(X_val)}

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    save_feature_columns(feature_list_path, feature_cols)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM from full_df Delta (holdout week excluded)")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-list-path", type=Path, default=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
    args, _ = parser.parse_known_args()

    train_df = load_train_df_table()
    print(f"training rows: {len(train_df):,} (holdout week excluded in Spark before collect)")
    metrics = train(train_df, model_path=args.model_path, feature_list_path=args.feature_list_path)
    print(f"Training complete: {metrics}")


if __name__ == "__main__":
    main()
