"""Shared Spark typing / columns for SparkXGBClassifier train and inference."""

from __future__ import annotations

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import BooleanType

from data.config import LABEL_COLUMN, XGB_VALIDATION_INDICATOR_COL


def cast_bools_to_double(sdf: DataFrame) -> DataFrame:
    out = sdf
    for col in out.columns:
        if isinstance(out.schema[col].dataType, BooleanType):
            out = out.withColumn(col, F.col(col).cast("double"))
    return out


def cast_label_double(sdf: DataFrame, label_col: str = LABEL_COLUMN) -> DataFrame:
    if label_col in sdf.columns:
        sdf = sdf.withColumn(label_col, F.col(label_col).cast("double"))
    return sdf


def prepare_spark_xgb_frame(sdf: DataFrame, *, label_col: str = LABEL_COLUMN) -> DataFrame:
    return cast_label_double(cast_bools_to_double(sdf), label_col=label_col)


def add_validation_indicator(sdf: DataFrame, *, is_validation: bool) -> DataFrame:
    return sdf.withColumn(XGB_VALIDATION_INDICATOR_COL, F.lit(bool(is_validation)))


def ensure_inference_validation_column(sdf: DataFrame) -> DataFrame:
    """Spark ML pipeline may expect the training-time indicator column at transform."""
    if XGB_VALIDATION_INDICATOR_COL in sdf.columns:
        return sdf.withColumn(XGB_VALIDATION_INDICATOR_COL, F.lit(False))
    return sdf.withColumn(XGB_VALIDATION_INDICATOR_COL, F.lit(False))


def estimate_scale_pos_weight(train_sdf: DataFrame, *, label_col: str = LABEL_COLUMN) -> float | None:
    counts = {int(r[label_col]): int(r["count"]) for r in train_sdf.groupBy(label_col).count().collect()}
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    if pos <= 0 or neg <= 0:
        return None
    return neg / pos
