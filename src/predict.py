from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from data.config import DEFAULT_FEATURE_LIST_PATH, DEFAULT_MODEL_PATH
from data.constants import full_df_table, inferences_table
from features import load_feature_columns, prepare_inference_features
from load_data import save_inferences

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
    parser.add_argument(
        "--input-table",
        default=full_df_table,
        help="Delta table with same schema as full_df (default: cas_search_reranking_full_df)",
    )
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-list-path", type=Path, default=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
    args = parser.parse_args()

    from spark_session import get_spark

    full_df = get_spark().table(args.input_table).toPandas()
    scores = predict_proba(full_df, model_path=args.model_path, feature_list_path=args.feature_list_path)
    out = full_df.copy()
    out["click_proba"] = scores
    save_inferences(out)
    print(f"Wrote {len(out):,} scored rows to {inferences_table}")


if __name__ == "__main__":
    main()
