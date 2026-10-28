from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from data.config import DEFAULT_FEATURE_LIST_PATH, DEFAULT_MODEL_PATH
from data.constants import inferences_table, schema
from features import load_feature_columns, prepare_inference_features
from load_data import load_holdout_df_table, save_inferences

REPO_ROOT = Path(__file__).resolve().parents[1]


def infer_holdout(
    holdout_df: pd.DataFrame,
    *,
    model_path: Path,
    feature_list_path: Path,
) -> None:
    booster = lgb.Booster(model_file=str(model_path))
    feature_columns = load_feature_columns(feature_list_path)
    X = prepare_inference_features(holdout_df, feature_columns)
    out = holdout_df.copy()
    out["click_proba"] = booster.predict(X)
    save_inferences(out)
    print(f"Wrote {len(out):,} scored holdout rows to {schema}.{inferences_table}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score holdout week with trained LightGBM (game_position nullified)")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-list-path", type=Path, default=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
    args, _ = parser.parse_known_args()

    holdout_df = load_holdout_df_table()
    infer_holdout(holdout_df, model_path=args.model_path, feature_list_path=args.feature_list_path)


if __name__ == "__main__":
    main()
