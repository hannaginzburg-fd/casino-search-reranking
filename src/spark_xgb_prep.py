"""Shared Spark typing / columns for SparkXGBClassifier train and inference."""

from __future__ import annotations

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import BooleanType

from data.config import INVERSE_PROPENSITY_WEIGHTING, LABEL_COLUMN, XGB_VALIDATION_INDICATOR_COL
from data.constants import sample_weight_col


def cast_bools_to_double(sdf: DataFrame) -> DataFrame:
    exprs = []
    for field in sdf.schema.fields:
        if isinstance(field.dataType, BooleanType):
            exprs.append(F.col(field.name).cast("double").alias(field.name))
        else:
            exprs.append(F.col(field.name))
    return sdf.select(*exprs)


def cast_label_double(sdf: DataFrame, label_col: str = LABEL_COLUMN) -> DataFrame:
    if label_col not in sdf.columns:
        return sdf
    passthrough = [F.col(c) for c in sdf.columns if c != label_col]
    passthrough.append(F.col(label_col).cast("double").alias(label_col))
    return sdf.select(*passthrough)


def prepare_spark_xgb_frame(sdf: DataFrame, *, label_col: str = LABEL_COLUMN) -> DataFrame:
    return cast_label_double(cast_bools_to_double(sdf), label_col=label_col)


def add_validation_indicator(sdf: DataFrame, *, is_validation: bool) -> DataFrame:
    lit = F.lit(bool(is_validation))
    if XGB_VALIDATION_INDICATOR_COL in sdf.columns:
        passthrough = [F.col(c) for c in sdf.columns if c != XGB_VALIDATION_INDICATOR_COL]
        passthrough.append(lit.alias(XGB_VALIDATION_INDICATOR_COL))
        return sdf.select(*passthrough)
    return sdf.withColumn(XGB_VALIDATION_INDICATOR_COL, lit)


def ensure_inference_validation_column(sdf: DataFrame) -> DataFrame:
    """Spark ML pipeline may expect the training-time indicator column at transform."""
    if XGB_VALIDATION_INDICATOR_COL in sdf.columns:
        passthrough = [F.col(c) for c in sdf.columns if c != XGB_VALIDATION_INDICATOR_COL]
        passthrough.append(F.lit(False).alias(XGB_VALIDATION_INDICATOR_COL))
        return sdf.select(*passthrough)
    return sdf.withColumn(XGB_VALIDATION_INDICATOR_COL, F.lit(False))


def prepare_spark_xgb_fit_union(
    train_sdf: DataFrame,
    val_sdf: DataFrame,
    feature_cols: list[str],
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Align train/validation schema (label, optional IPW weight, early-stop flag) and union for fit."""
    if LABEL_COLUMN not in train_sdf.columns:
        raise ValueError(f"Train frame missing {LABEL_COLUMN!r}")
    if LABEL_COLUMN not in val_sdf.columns:
        raise ValueError(f"Validation frame missing {LABEL_COLUMN!r} (required for Spark XGB early stopping)")

    train_sdf = cast_label_double(cast_bools_to_double(train_sdf))
    val_sdf = cast_label_double(cast_bools_to_double(val_sdf))

    if INVERSE_PROPENSITY_WEIGHTING:
        if sample_weight_col not in train_sdf.columns:
            raise ValueError(f"IPW enabled but train frame missing {sample_weight_col!r}")
    else:
        if sample_weight_col in train_sdf.columns:
            train_sdf = train_sdf.drop(sample_weight_col)
        if sample_weight_col in val_sdf.columns:
            val_sdf = val_sdf.drop(sample_weight_col)

    train_sdf = add_validation_indicator(train_sdf, is_validation=False)
    val_sdf = add_validation_indicator(val_sdf, is_validation=True)
    if INVERSE_PROPENSITY_WEIGHTING and sample_weight_col not in val_sdf.columns:
        val_sdf = val_sdf.withColumn(sample_weight_col, F.lit(1.0))

    fit_cols = [*feature_cols, LABEL_COLUMN, XGB_VALIDATION_INDICATOR_COL]
    if INVERSE_PROPENSITY_WEIGHTING:
        fit_cols.append(sample_weight_col)

    missing_train = [c for c in fit_cols if c not in train_sdf.columns]
    missing_val = [c for c in fit_cols if c not in val_sdf.columns]
    if missing_train or missing_val:
        raise ValueError(
            "Spark XGB fit frame missing columns: "
            f"train={missing_train or None}, validation={missing_val or None}"
        )

    train_sdf = train_sdf.select(*fit_cols)
    val_sdf = val_sdf.select(*fit_cols)
    return train_sdf, val_sdf, train_sdf.unionByName(val_sdf)


def prepare_spark_inference_input(sdf: DataFrame) -> DataFrame:
    """Columns SparkXGB pipelines expect at transform (all position-bias / IPW modes)."""
    out = ensure_inference_validation_column(cast_bools_to_double(sdf))
    # Pipelines fit with weight_col still require this column at transform (value unused).
    if sample_weight_col not in out.columns:
        passthrough = [F.col(c) for c in out.columns]
        passthrough.append(F.lit(1.0).alias(sample_weight_col))
        out = out.select(*passthrough)
    return out


def estimate_scale_pos_weight(train_sdf: DataFrame, *, label_col: str = LABEL_COLUMN) -> float | None:
    counts = {int(r[label_col]): int(r["count"]) for r in train_sdf.groupBy(label_col).count().collect()}
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    if pos <= 0 or neg <= 0:
        return None
    return neg / pos
