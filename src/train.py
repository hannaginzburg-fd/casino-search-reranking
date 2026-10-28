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
from sklearn.model_selection import train_test_split
from xgboost.spark import SparkXGBClassifier

from data.config import (
    DEFAULT_DRIVER_MODEL_PATH,
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_SPARK_MODEL_PATH,
    LABEL_COLUMN,
    XGB_PREDICTION_COL,
    XGB_PROBABILITY_COL,
)
from data.constants import driver_model_dbfs_path, spark_xgb_training, xgb_barrier_mode, xgb_num_workers
from features import save_feature_columns, training_matrix
from load_data import load_train_df_table, load_train_spark_df, save_train_df, save_train_spark_df
from spark_session import get_spark

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    sdf = load_train_spark_df()
    feature_cols = [c for c in sdf.columns if c != LABEL_COLUMN]
    save_feature_columns(feature_list_path, feature_cols)
    save_train_spark_df(sdf.select(*feature_cols, LABEL_COLUMN))

    sdf, prep_stages = _prep_spark_features(sdf, feature_cols)
    train_sdf, val_sdf = sdf.randomSplit([0.8, 0.2], seed=random_state)
    print(
        f"SparkXGBClassifier num_workers={xgb_num_workers}, barrier_mode={xgb_barrier_mode} "
        f"(train rows ~{train_sdf.count():,}) — requires single-user / assigned cluster"
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
    pdf = load_train_df_table()
    X, y, feature_cols = training_matrix(pdf)
    save_feature_columns(feature_list_path, feature_cols)
    save_train_df(X.assign(**{LABEL_COLUMN: y}))

    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype) == "string":
            X[col] = X[col].astype("category")

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )

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
    parser = argparse.ArgumentParser(description="Train XGBoost from full_df Delta (holdout week excluded)")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--feature-list-path", type=Path, default=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
    args, _ = parser.parse_known_args()

    get_spark()
    if spark_xgb_training:
        model_path = args.model_path or (REPO_ROOT / DEFAULT_SPARK_MODEL_PATH)
        metrics = train_spark(model_path=model_path, feature_list_path=args.feature_list_path)
    else:
        model_path = args.model_path or (REPO_ROOT / DEFAULT_DRIVER_MODEL_PATH)
        metrics = train_driver(model_path=model_path, feature_list_path=args.feature_list_path)

    print(f"Training complete: {metrics}")


if __name__ == "__main__":
    main()
