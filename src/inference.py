from __future__ import annotations

import argparse
from pathlib import Path

import xgboost as xgb
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import DoubleType, StructField, StructType
from data.config import (
    DEFAULT_DRIVER_MODEL_PATH,
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_SPARK_MODEL_PATH,
    INFERENCE_DELTA_CASTS,
    INFERENCE_OUTPUT_COLUMNS,
    INVERSE_PROPENSITY_WEIGHTING,
    XGB_PROBABILITY_COL,
    validate_pipeline_config,
)
from data.constants import driver_model_dbfs_path, inferences_table, schema, spark_xgb_training
from features import assert_frame_has_columns, load_feature_columns, prepare_inference_features
from inference_leakage import apply_leakage_fill_for_scoring, compute_leakage_imputation_values
from load_data import load_holdout_spark_df, save_inferences_spark
from artifact_utils import reset_inference_artifacts
from holdout_eval import print_holdout_ranking_diagnostics
from model_diagnostics import print_xgb_model_diagnostics
from spark_session import get_spark
from pyspark.sql import functions as F
from spark_xgb_prep import cast_bools_to_double, prepare_spark_inference_input

REPO_ROOT = Path(__file__).resolve().parents[1]


def _select_inference_output(scored, output_cols: list[str]):
    present = [c for c in output_cols if c in scored.columns]
    missing = [c for c in output_cols if c not in scored.columns]
    if missing:
        raise ValueError(f"Inference output missing columns: {missing}")
    out = scored.select(*present, "click_proba")
    for col, spark_type in INFERENCE_DELTA_CASTS.items():
        if col in out.columns:
            out = out.withColumn(col, F.col(col).cast(spark_type))
    return out


def _prepare_model_input(holdout, feature_cols: list[str]):
    if INVERSE_PROPENSITY_WEIGHTING:
        return cast_bools_to_double(holdout), None
    imputation = compute_leakage_imputation_values(holdout, feature_cols)
    from data.config import INFERENCE_IMPUTE_LEAKAGE_COLS

    if INFERENCE_IMPUTE_LEAKAGE_COLS:
        print(f"Inference leakage imputation (batch mean/mode): {imputation}")
    model_input = apply_leakage_fill_for_scoring(holdout, feature_cols, imputation_values=imputation)
    return cast_bools_to_double(model_input), imputation


def score_holdout_spark(*, model_path: Path, feature_list_path: Path):
    holdout = load_holdout_spark_df().withColumn("__row_id", F.monotonically_increasing_id())
    feature_cols = load_feature_columns(feature_list_path)
    if not feature_cols:
        raise ValueError("feature_columns.json is empty; re-run training")
    assert_frame_has_columns(list(holdout.columns), feature_cols, frame_name="Holdout")
    model_input, _ = _prepare_model_input(holdout, feature_cols)
    model_input = prepare_spark_inference_input(model_input)

    pipeline = PipelineModel.load(str(model_path))
    scored = pipeline.transform(model_input)
    probs = scored.select(
        "__row_id",
        vector_to_array(XGB_PROBABILITY_COL).getItem(1).alias("click_proba"),
    )
    return holdout.join(probs, on="__row_id", how="inner").drop("__row_id")


def infer_holdout_spark(*, model_path: Path, feature_list_path: Path) -> None:
    out = score_holdout_spark(model_path=model_path, feature_list_path=feature_list_path)
    save_inferences_spark(_select_inference_output(out, INFERENCE_OUTPUT_COLUMNS))
    print(f"Appended scored holdout rows to {schema}.{inferences_table} (inserted_ts on each row)")


def score_holdout_driver(*, model_path: Path, feature_list_path: Path):
    feature_cols = load_feature_columns(feature_list_path)
    if not feature_cols:
        raise ValueError("feature_columns.json is empty; re-run training")
    holdout = load_holdout_spark_df().withColumn("__row_id", F.monotonically_increasing_id())
    assert_frame_has_columns(list(holdout.columns), feature_cols, frame_name="Holdout")
    model_input, imputation = _prepare_model_input(holdout, feature_cols)

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
            X = prepare_inference_features(pdf, feature_cols, leakage_imputation=imputation)
            for col in X.columns:
                if X[col].dtype == "object" or str(X[col].dtype) == "string":
                    X[col] = X[col].astype("category")
            out = pdf[["__row_id"]].copy()
            out["click_proba"] = booster.predict(xgb.DMatrix(X, enable_categorical=True))
            yield out

    probs = model_input.mapInPandas(score_partition, schema=prob_schema)
    return holdout.join(probs, on="__row_id", how="inner").drop("__row_id")


def infer_holdout_driver(*, model_path: Path, feature_list_path: Path) -> None:
    out = score_holdout_driver(model_path=model_path, feature_list_path=feature_list_path)
    save_inferences_spark(_select_inference_output(out, INFERENCE_OUTPUT_COLUMNS))
    print(f"Appended scored holdout rows to {schema}.{inferences_table} (driver model, inserted_ts on each row)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score holdout week (leakage cols imputed or nulled on scoring copy)")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--feature-list-path", type=Path, default=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
    args, _ = parser.parse_known_args()

    validate_pipeline_config()
    reset_inference_artifacts()
    if spark_xgb_training:
        get_spark()
        model_path = args.model_path or (REPO_ROOT / DEFAULT_SPARK_MODEL_PATH)
        feature_list_path = args.feature_list_path or (REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
        print_xgb_model_diagnostics(model_path=model_path, feature_list_path=feature_list_path, use_spark_pipeline=True)
        scored = score_holdout_spark(model_path=model_path, feature_list_path=feature_list_path)
        save_inferences_spark(_select_inference_output(scored, INFERENCE_OUTPUT_COLUMNS))
        print(f"Appended scored holdout rows to {schema}.{inferences_table} (inserted_ts on each row)")
        print_holdout_ranking_diagnostics(scored)
    else:
        get_spark()
        model_path = args.model_path or (REPO_ROOT / DEFAULT_DRIVER_MODEL_PATH)
        feature_list_path = args.feature_list_path or (REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
        print_xgb_model_diagnostics(model_path=model_path, feature_list_path=feature_list_path, use_spark_pipeline=False)
        scored = score_holdout_driver(model_path=model_path, feature_list_path=feature_list_path)
        save_inferences_spark(_select_inference_output(scored, INFERENCE_OUTPUT_COLUMNS))
        print(f"Appended scored holdout rows to {schema}.{inferences_table} (driver model, inserted_ts on each row)")
        print_holdout_ranking_diagnostics(scored)


if __name__ == "__main__":
    main()
