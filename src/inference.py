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
    XGB_PROBABILITY_COL,
)
from data.constants import driver_model_dbfs_path, inferences_table, schema, spark_xgb_training
from features import load_feature_columns, prepare_inference_features
from load_data import load_holdout_spark_df, save_inferences_spark

REPO_ROOT = Path(__file__).resolve().parents[1]


def infer_holdout_spark(*, model_path: Path) -> None:
    holdout = load_holdout_spark_df()
    for col in INFERENCE_NULL_COLUMNS:
        if col in holdout.columns:
            holdout = holdout.withColumn(col, F.lit(None).cast("double"))
    for col in holdout.columns:
        if isinstance(holdout.schema[col].dataType, BooleanType):
            holdout = holdout.withColumn(col, F.col(col).cast("double"))

    base_cols = holdout.columns
    pipeline = PipelineModel.load(str(model_path))
    scored = pipeline.transform(holdout)
    scored = scored.withColumn("click_proba", vector_to_array(XGB_PROBABILITY_COL).getItem(1))
    save_inferences_spark(scored.select(*base_cols, "click_proba"))
    print(f"Wrote scored holdout rows to {schema}.{inferences_table}")


def infer_holdout_driver(*, model_path: Path, feature_list_path: Path) -> None:
    feature_cols = load_feature_columns(feature_list_path)
    holdout = load_holdout_spark_df()
    for col in INFERENCE_NULL_COLUMNS:
        if col in holdout.columns:
            holdout = holdout.withColumn(col, F.lit(None).cast("double"))
    for col in holdout.columns:
        if isinstance(holdout.schema[col].dataType, BooleanType):
            holdout = holdout.withColumn(col, F.col(col).cast("double"))

    base_cols = holdout.columns
    schema_fields = [StructField(c, holdout.schema[c].dataType, True) for c in base_cols]
    schema_fields.append(StructField("click_proba", DoubleType(), True))
    out_schema = StructType(schema_fields)
    model_file = driver_model_dbfs_path if Path(driver_model_dbfs_path).exists() else str(model_path)

    def score_partition(iterator):
        booster = xgb.Booster()
        booster.load_model(model_file)
        for pdf in iterator:
            X = prepare_inference_features(pdf, feature_cols)
            for col in X.columns:
                if X[col].dtype == "object" or str(X[col].dtype) == "string":
                    X[col] = X[col].astype("category")
            pdf = pdf.copy()
            pdf["click_proba"] = booster.predict(xgb.DMatrix(X, enable_categorical=True))
            yield pdf[list(base_cols) + ["click_proba"]]

    scored = holdout.mapInPandas(score_partition, schema=out_schema)
    save_inferences_spark(scored)
    print(f"Wrote scored holdout rows to {schema}.{inferences_table} (driver model, Spark write)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score holdout week (game_position nullified)")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--feature-list-path", type=Path, default=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
    args, _ = parser.parse_known_args()

    if spark_xgb_training:
        model_path = args.model_path or (REPO_ROOT / DEFAULT_SPARK_MODEL_PATH)
        infer_holdout_spark(model_path=model_path)
    else:
        model_path = args.model_path or (REPO_ROOT / DEFAULT_DRIVER_MODEL_PATH)
        infer_holdout_driver(model_path=model_path, feature_list_path=args.feature_list_path)


if __name__ == "__main__":
    main()
