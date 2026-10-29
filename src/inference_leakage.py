"""Serve-time handling for SCORE_OR_RANK_LEAKAGE_COLS on the scoring copy (not persisted holdout)."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, functions as F
from pyspark.sql.types import BooleanType, NumericType, StringType

from data.config import INFERENCE_IMPUTE_LEAKAGE_COLS, INFERENCE_NULL_COLUMNS, INVERSE_PROPENSITY_WEIGHTING


def _leakage_cols_for_model(feature_cols: list[str], sdf: DataFrame) -> list[str]:
    return [c for c in INFERENCE_NULL_COLUMNS if c in feature_cols and c in sdf.columns]


def _spark_mode_value(sdf: DataFrame, col: str):
    row = (
        sdf.filter(F.col(col).isNotNull())
        .groupBy(col)
        .count()
        .orderBy(F.desc("count"))
        .first()
    )
    return None if row is None else row[col]


def compute_leakage_imputation_values(sdf: DataFrame, feature_cols: list[str]) -> dict[str, object]:
    """Mean for numeric columns, mode for string/boolean (computed on the batch being scored)."""
    values: dict[str, object] = {}
    for col in _leakage_cols_for_model(feature_cols, sdf):
        dt = sdf.schema[col].dataType
        if isinstance(dt, StringType):
            values[col] = _spark_mode_value(sdf, col)
        elif isinstance(dt, BooleanType):
            mode = _spark_mode_value(sdf.withColumn(col, F.col(col).cast("string")), col)
            values[col] = mode == "true" if mode is not None else None
        elif isinstance(dt, NumericType):
            values[col] = sdf.select(F.avg(col).alias("v")).first()["v"]
        else:
            values[col] = _spark_mode_value(sdf.withColumn(col, F.col(col).cast("string")), col)
    return values


def _fill_literal(value: object, spark_type) -> Column:
    if value is None:
        return F.lit(None).cast(spark_type)
    if isinstance(value, bool):
        return F.lit(value).cast(spark_type)
    if isinstance(value, (int, float)):
        return F.lit(value).cast(spark_type)
    return F.lit(str(value)).cast(spark_type)


def _replace_column(sdf: DataFrame, col: str, expr: Column) -> DataFrame:
    return sdf.select(*[expr.alias(col) if c == col else F.col(c) for c in sdf.columns])


def apply_leakage_fill_for_scoring(
    sdf: DataFrame,
    feature_cols: list[str],
    *,
    imputation_values: dict[str, object] | None = None,
) -> DataFrame:
    """Mutate leakage feature columns on the scoring copy only (null or mean/mode imputation)."""
    if INVERSE_PROPENSITY_WEIGHTING:
        return sdf
    cols = _leakage_cols_for_model(feature_cols, sdf)
    if not cols:
        return sdf

    out = sdf
    if INFERENCE_IMPUTE_LEAKAGE_COLS:
        imputation_values = imputation_values or compute_leakage_imputation_values(out, feature_cols)
        for col in cols:
            fill = imputation_values.get(col)
            out = _replace_column(out, col, _fill_literal(fill, out.schema[col].dataType))
        return out

    for col in cols:
        out = _replace_column(out, col, F.lit(None).cast(out.schema[col].dataType))
    return out
