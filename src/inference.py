from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from data.config import (
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SPARK_MODEL_PATH,
    INFERENCE_NULL_COLUMNS,
)
from data.constants import inferences_table, schema, use_synapseml
from features import load_feature_columns, prepare_inference_features
from load_data import load_holdout_df_table, load_holdout_spark_df, save_inferences, save_inferences_spark

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


def infer_holdout_on_spark(*, model_path: Path) -> None:
    from pyspark.ml import PipelineModel

    holdout = load_holdout_spark_df()
    for col in INFERENCE_NULL_COLUMNS:
        if col in holdout.columns:
            holdout = holdout.withColumn(col, F.lit(None))

    base_cols = holdout.columns
    pipeline = PipelineModel.load(str(model_path))
    scored = pipeline.transform(holdout)
    scored = scored.withColumn("click_proba", vector_to_array("probability").getItem(1))
    save_inferences_spark(scored.select(*base_cols, "click_proba"))
    print(f"Wrote scored holdout rows to {schema}.{inferences_table} (Spark, no driver collect)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score holdout week with trained LightGBM (game_position nullified)")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--feature-list-path", type=Path, default=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
    args, _ = parser.parse_known_args()

    if use_synapseml:
        model_path = args.model_path or (REPO_ROOT / DEFAULT_SPARK_MODEL_PATH)
        infer_holdout_on_spark(model_path=model_path)
    else:
        model_path = args.model_path or (REPO_ROOT / DEFAULT_MODEL_PATH)
        holdout_df = load_holdout_df_table()
        infer_holdout(holdout_df, model_path=model_path, feature_list_path=args.feature_list_path)


if __name__ == "__main__":
    main()
