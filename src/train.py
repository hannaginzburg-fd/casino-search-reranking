from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from config import DEFAULT_FEATURE_LIST_PATH, DEFAULT_FULL_DF_PATH, DEFAULT_MODEL_PATH
from features import save_feature_columns, training_matrix
from load_data import load_full_df, load_full_df_parquet, save_full_df

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQL = REPO_ROOT / "sql" / "training_data.sql"


def train(
    full_df: pd.DataFrame,
    *,
    model_path: Path,
    feature_list_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    X, y, feature_cols = training_matrix(full_df)
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
    parser = argparse.ArgumentParser(description="Train search click LightGBM from Databricks feature SQL")
    parser.add_argument("--sql", type=Path, default=DEFAULT_SQL, help="Path to training_data.sql")
    parser.add_argument("--from-parquet", type=Path, default=None, help="Skip SQL; load full_df from parquet")
    parser.add_argument("--save-parquet", type=Path, default=DEFAULT_FULL_DF_PATH, help="Cache full_df after SQL")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-list-path", type=Path, default=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
    args = parser.parse_args()

    if args.from_parquet:
        full_df = load_full_df_parquet(args.from_parquet)
    else:
        full_df = load_full_df(args.sql)
        if args.save_parquet:
            save_full_df(full_df, args.save_parquet)

    print(f"full_df shape: {full_df.shape}")
    metrics = train(full_df, model_path=args.model_path, feature_list_path=args.feature_list_path)
    print(f"Training complete: {metrics}")


if __name__ == "__main__":
    main()
