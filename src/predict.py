from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from config import DEFAULT_FEATURE_LIST_PATH, DEFAULT_MODEL_PATH
from features import load_feature_columns, prepare_inference_features

REPO_ROOT = Path(__file__).resolve().parents[1]


def predict_proba(
    full_df: pd.DataFrame,
    *,
    model_path: Path,
    feature_list_path: Path,
) -> pd.Series:
    booster = lgb.Booster(model_file=str(model_path))
    feature_columns = load_feature_columns(feature_list_path)
    X = prepare_inference_features(full_df, feature_columns)
    return pd.Series(booster.predict(X), index=full_df.index, name="click_proba")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score rows with trained LightGBM (game_position nullified)")
    parser.add_argument("--input-parquet", type=Path, required=True, help="Rows with same schema as full_df")
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-list-path", type=Path, default=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
    args = parser.parse_args()

    full_df = pd.read_parquet(args.input_parquet)
    scores = predict_proba(full_df, model_path=args.model_path, feature_list_path=args.feature_list_path)
    out = full_df.copy()
    out["click_proba"] = scores
    out.to_parquet(args.output_parquet, index=False)
    print(f"Wrote {len(out):,} scored rows to {args.output_parquet}")


if __name__ == "__main__":
    main()
