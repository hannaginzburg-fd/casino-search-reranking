from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import xgboost as xgb
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import functions as F
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
    share_df_summaries,
)
from df_summaries import run_training_summaries
from data.constants import driver_model_dbfs_path, spark_xgb_training, xgb_barrier_mode, xgb_num_workers
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_OUTPUT_DIR = REPO_ROOT / "artifacts" / "training_summaries"


def _maybe_share_df_summaries() -> None:
    if not share_df_summaries:
        return
    run_training_summaries(load_train_summary_spark_df(), output_dir=SUMMARY_OUTPUT_DIR)


def _prep_spark_features(sdf, feature_cols):
    for col in feature_cols:
        if isinstance(sdf.schema[col].dataType, BooleanType):
            sdf = sdf.withColumn(col, F.col(col).cast("double"))
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
    for col in feature_cols:
        if isinstance(val_sdf.schema[col].dataType, BooleanType):
            val_sdf = val_sdf.withColumn(col, F.col(col).cast("double"))
    print(
        f"SparkXGBClassifier num_workers={xgb_num_workers}, barrier_mode={xgb_barrier_mode} "
        f"(train rows ~{train_sdf.count():,}, validation rows ~{val_sdf.count():,}) "
        f"— requires single-user / assigned cluster"
    )

    xgb_estimator = SparkXGBClassifier(
        features_col="features",
        label_col=LABEL_COLUMN,
        prediction_col=XGB_PREDICTION_COL,
        probability_col=XGB_PROBABILITY_COL,
        num_workers=xgb_num_workers,
        force_repartition=False,
        use_gpu=False,
        barrier_mode=xgb_barrier_mode,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
        max_depth=8,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    pipeline = Pipeline(stages=prep_stages + [xgb_estimator])
    train_sdf.cache()
    train_rows = train_sdf.count()
    model = pipeline.fit(train_sdf)
    train_sdf.unpersist()

    val_pred = model.transform(val_sdf)
    val_auc = BinaryClassificationEvaluator(
        labelCol=LABEL_COLUMN,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(val_pred)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str(model_path))
    return {"val_auc": float(val_auc), "train_rows": train_rows, "val_rows": val_sdf.count()}


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

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=8,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        enable_categorical=True,
        tree_method="hist",
    )
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
