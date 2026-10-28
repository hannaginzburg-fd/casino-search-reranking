from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, StringType
from xgboost.spark import SparkXGBClassifier

from data.config import (
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_MODEL_PATH,
    LABEL_COLUMN,
    XGB_PREDICTION_COL,
    XGB_PROBABILITY_COL,
)
from features import save_feature_columns
from load_data import load_train_spark_df, save_train_spark_df
from spark_session import get_spark

REPO_ROOT = Path(__file__).resolve().parents[1]


def train(*, model_path: Path, feature_list_path: Path, random_state: int = 42) -> dict[str, float]:
    sdf = load_train_spark_df()
    feature_cols = [c for c in sdf.columns if c != LABEL_COLUMN]
    save_feature_columns(feature_list_path, feature_cols)
    save_train_spark_df(sdf.select(*feature_cols, LABEL_COLUMN))

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

    train_sdf, val_sdf = sdf.randomSplit([0.8, 0.2], seed=random_state)
    row_count = train_sdf.count()
    data_size_gb = row_count * len(sdf.columns) * 8 / (1024**3)
    num_workers = max(2, min(8, int(data_size_gb) or 2))
    print(f"SparkXGBClassifier num_workers={num_workers} (train rows ~{row_count:,})")

    xgb = SparkXGBClassifier(
        features_col="features",
        label_col=LABEL_COLUMN,
        prediction_col=XGB_PREDICTION_COL,
        probability_col=XGB_PROBABILITY_COL,
        num_workers=num_workers,
        force_repartition=False,
        use_gpu=False,
        barrier_mode=True,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
        max_depth=8,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    pipeline = Pipeline(stages=prep_stages + [xgb])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spark XGBoost from full_df Delta (holdout week excluded)")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-list-path", type=Path, default=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
    args, _ = parser.parse_known_args()

    get_spark()
    metrics = train(model_path=args.model_path, feature_list_path=args.feature_list_path)
    print(f"Training complete: {metrics}")


if __name__ == "__main__":
    main()
