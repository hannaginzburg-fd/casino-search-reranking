from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import xgboost as xgb
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import BooleanType, StringType
from sklearn.metrics import roc_auc_score
from xgboost.spark import SparkXGBClassifier

from data.config import (
    DEFAULT_DRIVER_MODEL_PATH,
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_SPARK_MODEL_PATH,
    LABEL_COLUMN,
    XGB_PREDICTION_COL,
    XGB_PROBABILITY_COL,
    XGB_VALIDATION_INDICATOR_COL,
    share_df_summaries,
)
from df_summaries import run_training_summaries
from data.constants import (
    driver_model_dbfs_path,
    spark_xgb_training,
    xgb_barrier_mode,
    xgb_auto_scale_pos_weight,
    xgb_early_stopping_rounds,
    xgb_n_estimators,
    xgb_num_workers,
)
from features import save_feature_columns, training_matrix
from load_data import (
    load_train_df_table,
    load_train_spark_df,
    load_train_summary_spark_df,
    load_validation_df_table,
    load_validation_spark_df,
    save_train_df,
    save_train_spark_df,
    save_validation_spark_df,
)
from spark_session import get_spark
from spark_xgb_prep import (
    add_validation_indicator,
    cast_bools_to_double,
    cast_label_double,
    estimate_scale_pos_weight,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_OUTPUT_DIR = REPO_ROOT / "artifacts" / "training_summaries"


def _spark_task_slots() -> int:
    spark = get_spark()
    conf = spark.sparkContext.getConf()
    instances = conf.get("spark.executor.instances")
    cores = int(conf.get("spark.executor.cores", "1") or "1")
    if instances and str(instances).isdigit():
        return max(1, int(instances) * cores)
    # Dynamic allocation / Databricks: use default parallelism as a conservative slot estimate
    return max(1, spark.sparkContext.defaultParallelism)


def _resolve_xgb_num_workers(train_sdf, *, requested: int | None) -> int:
    want = requested if requested is not None else 8
    slots = _spark_task_slots()
    parts = max(1, train_sdf.rdd.getNumPartitions())
    effective = max(1, min(want, slots, parts))
    if effective != want:
        print(
            f"Adjusted xgb num_workers {want} -> {effective} "
            f"(task_slots~{slots}, train_partitions={parts})"
        )
    if xgb_barrier_mode and effective < 2:
        print(
            "Warning: barrier_mode with num_workers=1 is fragile; "
            "set xgb_barrier_mode=False or add executors."
        )
    return effective


def _maybe_share_df_summaries() -> None:
    if not share_df_summaries:
        return
    run_training_summaries(load_train_summary_spark_df(), output_dir=SUMMARY_OUTPUT_DIR)


def _prep_spark_features(sdf, feature_cols):
    sdf = cast_bools_to_double(sdf)
    prep_stages = []
    vec_cols = []
    for col in feature_cols:
        if isinstance(sdf.schema[col].dataType, StringType):
            indexed = f"__idx_{col}"
            prep_stages.append(StringIndexer(inputCol=col, outputCol=indexed, handleInvalid="keep"))
            vec_cols.append(indexed)
        else:
            vec_cols.append(col)
    prep_stages.append(VectorAssembler(inputCols=vec_cols, outputCol="features", handleInvalid="keep"))
    return sdf, prep_stages


def train_spark(*, model_path: Path, feature_list_path: Path, random_state: int = 42) -> dict[str, float]:
    train_sdf = load_train_spark_df()
    val_sdf = load_validation_spark_df()
    feature_cols = [c for c in train_sdf.columns if c != LABEL_COLUMN]
    save_feature_columns(feature_list_path, feature_cols)
    save_train_spark_df(train_sdf.select(*feature_cols, LABEL_COLUMN))
    save_validation_spark_df(val_sdf.select(*feature_cols, LABEL_COLUMN))

    train_sdf, prep_stages = _prep_spark_features(train_sdf, feature_cols)
    val_sdf = cast_bools_to_double(val_sdf)
    train_sdf = cast_label_double(train_sdf)
    val_sdf = cast_label_double(val_sdf)

    train_sdf = add_validation_indicator(train_sdf, is_validation=False)
    val_sdf = add_validation_indicator(val_sdf, is_validation=True)
    fit_sdf = train_sdf.unionByName(val_sdf)

    num_workers = _resolve_xgb_num_workers(fit_sdf, requested=xgb_num_workers)
    train_rows = train_sdf.count()
    val_rows = val_sdf.count()
    scale_pos_weight = None
    if xgb_auto_scale_pos_weight:
        scale_pos_weight = estimate_scale_pos_weight(train_sdf)
        if scale_pos_weight is not None:
            print(f"Spark XGB scale_pos_weight={scale_pos_weight:.4f} (neg/pos on train)")
    print(
        f"SparkXGBClassifier num_workers={num_workers}, barrier_mode={xgb_barrier_mode}, "
        f"n_estimators<={xgb_n_estimators}, early_stopping_rounds={xgb_early_stopping_rounds} "
        f"(train rows ~{train_rows:,}, validation rows ~{val_rows:,}) "
        f"— use on-demand executors; spot preemption aborts barrier training"
    )

    xgb_kwargs: dict = {
        "features_col": "features",
        "label_col": LABEL_COLUMN,
        "prediction_col": XGB_PREDICTION_COL,
        "probability_col": XGB_PROBABILITY_COL,
        "validation_indicator_col": XGB_VALIDATION_INDICATOR_COL,
        "num_workers": num_workers,
        "force_repartition": True,
        "use_gpu": False,
        "barrier_mode": xgb_barrier_mode,
        "eval_metric": "logloss",
        "random_state": random_state,
        "verbosity": 0,
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": xgb_n_estimators,
        "early_stopping_rounds": xgb_early_stopping_rounds,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    if scale_pos_weight is not None:
        xgb_kwargs["scale_pos_weight"] = scale_pos_weight
    xgb_estimator = SparkXGBClassifier(**xgb_kwargs)

    pipeline = Pipeline(stages=prep_stages + [xgb_estimator])
    fit_sdf.cache()
    model = pipeline.fit(fit_sdf)
    fit_sdf.unpersist()

    val_pred = model.transform(val_sdf)
    val_pred = val_pred.withColumn("click_proba", vector_to_array(XGB_PROBABILITY_COL).getItem(1))
    val_auc = BinaryClassificationEvaluator(
        labelCol=LABEL_COLUMN,
        probabilityCol="click_proba",
        metricName="areaUnderROC",
    ).evaluate(val_pred)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str(model_path))
    return {"val_auc": float(val_auc), "train_rows": train_rows, "val_rows": val_rows}


def train_driver(*, model_path: Path, feature_list_path: Path, random_state: int = 42) -> dict[str, float]:
    train_pdf = load_train_df_table()
    val_pdf = load_validation_df_table()
    X_train, y_train, feature_cols = training_matrix(train_pdf)
    X_val, y_val, _ = training_matrix(val_pdf)
    save_feature_columns(feature_list_path, feature_cols)
    save_train_df(X_train.assign(**{LABEL_COLUMN: y_train}))
    save_validation_spark_df(
        get_spark().createDataFrame(X_val.assign(**{LABEL_COLUMN: y_val}))
    )

    for frame in (X_train, X_val):
        for col in frame.columns:
            if frame[col].dtype == "object" or str(frame[col].dtype) == "string":
                frame[col] = frame[col].astype("category")

    driver_xgb_kwargs: dict = {
        "objective": "binary:logistic",
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": xgb_n_estimators,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "enable_categorical": True,
        "tree_method": "hist",
        "early_stopping_rounds": xgb_early_stopping_rounds,
    }
    if xgb_auto_scale_pos_weight:
        pos = int(y_train.sum())
        neg = len(y_train) - pos
        if pos > 0 and neg > 0:
            driver_xgb_kwargs["scale_pos_weight"] = neg / pos
            print(f"Driver XGB scale_pos_weight={driver_xgb_kwargs['scale_pos_weight']:.4f}")
    model = xgb.XGBClassifier(**driver_xgb_kwargs)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    dbfs_path = Path(driver_model_dbfs_path)
    if dbfs_path.parent.exists():
        dbfs_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_path, dbfs_path)
        print(f"Copied driver model to {driver_model_dbfs_path} for distributed inference")
    return {"val_auc": float(val_auc), "train_rows": len(X_train), "val_rows": len(X_val)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost from full_df Delta (temporal validation + holdout excluded from fit)"
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--feature-list-path", type=Path, default=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
    args, _ = parser.parse_known_args()

    get_spark()
    _maybe_share_df_summaries()
    if spark_xgb_training:
        model_path = args.model_path or (REPO_ROOT / DEFAULT_SPARK_MODEL_PATH)
        metrics = train_spark(model_path=model_path, feature_list_path=args.feature_list_path)
    else:
        model_path = args.model_path or (REPO_ROOT / DEFAULT_DRIVER_MODEL_PATH)
        metrics = train_driver(model_path=model_path, feature_list_path=args.feature_list_path)

    print(f"Training complete: {metrics}")


if __name__ == "__main__":
    main()
