from __future__ import annotations

import argparse
from pathlib import Path

import xgboost as xgb
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, StructField, StructType
from data.config import (
    DEFAULT_DRIVER_MODEL_PATH,
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_SPARK_MODEL_PATH,
    INFERENCE_NULL_COLUMNS,
    INFERENCE_OUTPUT_COLUMNS,
    XGB_PROBABILITY_COL,
)
from data.constants import driver_model_dbfs_path, inferences_table, schema, spark_xgb_training
from features import load_feature_columns, prepare_inference_features
from load_data import load_holdout_spark_df, save_inferences_spark

REPO_ROOT = Path(__file__).resolve().parents[1]


def _null_columns_for_scoring(sdf, feature_cols: list[str]):
    """Return a copy with serve-time-unknown columns nulled only when they are model features."""
    out = sdf
    for col in INFERENCE_NULL_COLUMNS:
        if col in feature_cols and col in out.columns:
            out = out.withColumn(col, F.lit(None).cast("double"))
    return out


def _cast_bools_to_double(sdf):
    out = sdf
    for col in out.columns:
        if isinstance(out.schema[col].dataType, BooleanType):
            out = out.withColumn(col, F.col(col).cast("double"))
    return out


def _select_inference_output(scored, output_cols: list[str]):
    present = [c for c in output_cols if c in scored.columns]
    missing = [c for c in output_cols if c not in scored.columns]
    if missing:
        raise ValueError(f"Inference output missing columns: {missing}")
    return scored.select(*present, "click_proba")


def infer_holdout_spark(*, model_path: Path, feature_list_path: Path) -> None:
    holdout = load_holdout_spark_df().withColumn("__row_id", F.monotonically_increasing_id())
    feature_cols = load_feature_columns(feature_list_path)
    model_input = _cast_bools_to_double(_null_columns_for_scoring(holdout, feature_cols))

    pipeline = PipelineModel.load(str(model_path))
    scored = pipeline.transform(model_input)
    probs = scored.select(
        "__row_id",
        vector_to_array(XGB_PROBABILITY_COL).getItem(1).alias("click_proba"),
    )
    out = holdout.join(probs, on="__row_id", how="inner").drop("__row_id")
    save_inferences_spark(_select_inference_output(out, INFERENCE_OUTPUT_COLUMNS))
    print(f"Wrote scored holdout rows to {schema}.{inferences_table}")


def infer_holdout_driver(*, model_path: Path, feature_list_path: Path) -> None:
    feature_cols = load_feature_columns(feature_list_path)
    holdout = load_holdout_spark_df().withColumn("__row_id", F.monotonically_increasing_id())
    model_input = _cast_bools_to_double(_null_columns_for_scoring(holdout, feature_cols))

    prob_schema = StructType(
        [
            StructField("__row_id", holdout.schema["__row_id"].dataType, False),
            StructField("click_proba", DoubleType(), True),
        ]
    )
    model_file = driver_model_dbfs_path if Path(driver_model_dbfs_path).exists() else str(model_path)

    def score_partition(iterator):
        booster = xgb.Booster()
        booster.load_model(model_file)
        for pdf in iterator:
            X = prepare_inference_features(pdf, feature_cols)
            for col in X.columns:
                if X[col].dtype == "object" or str(X[col].dtype) == "string":
                    X[col] = X[col].astype("category")
            out = pdf[["__row_id"]].copy()
            out["click_proba"] = booster.predict(xgb.DMatrix(X, enable_categorical=True))
            yield out

    probs = model_input.mapInPandas(score_partition, schema=prob_schema)
    out = holdout.join(probs, on="__row_id", how="inner").drop("__row_id")
    save_inferences_spark(_select_inference_output(out, INFERENCE_OUTPUT_COLUMNS))
    print(f"Wrote scored holdout rows to {schema}.{inferences_table} (driver model, Spark write)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score holdout week (game_position nullified)")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--feature-list-path", type=Path, default=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
    args, _ = parser.parse_known_args()

    if spark_xgb_training:
        model_path = args.model_path or (REPO_ROOT / DEFAULT_SPARK_MODEL_PATH)
        infer_holdout_spark(model_path=model_path, feature_list_path=args.feature_list_path)
    else:
        model_path = args.model_path or (REPO_ROOT / DEFAULT_DRIVER_MODEL_PATH)
        infer_holdout_driver(model_path=model_path, feature_list_path=args.feature_list_path)


if __name__ == "__main__":
    main()
