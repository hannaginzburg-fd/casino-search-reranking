"""Inverse propensity weights for position-bias correction (training only)."""

from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from data.constants import ipw_weight_clip_max


def add_inverse_propensity_weights(
    sdf: DataFrame,
    *,
    position_col: str,
    weight_col: str,
) -> DataFrame:
    """
    Weight = 1 / P(position) estimated by marginal position share on the train slice (clipped).
    """
    if position_col not in sdf.columns:
        raise ValueError(f"IPW requires position column {position_col!r} on the train frame")

    total = sdf.count()
    props = (
        sdf.filter(F.col(position_col).isNotNull())
        .groupBy(position_col)
        .count()
        .withColumn("propensity", F.col("count") / F.lit(float(total)))
    )
    weighted = sdf.join(props.select(position_col, "propensity"), on=position_col, how="left")
    weighted = weighted.withColumn(
        weight_col,
        F.when(
            F.col("propensity").isNull() | (F.col("propensity") <= 0),
            F.lit(1.0),
        ).otherwise(F.least(F.lit(float(ipw_weight_clip_max)), 1.0 / F.col("propensity"))),
    )
    return weighted.drop("propensity")


def ipw_weights_pandas(series, *, clip_max: float | None = None):
    """Driver-path IPW from a pandas position column."""
    clip = clip_max if clip_max is not None else ipw_weight_clip_max
    counts = series.value_counts(dropna=False)
    total = len(series)
    prop = series.map(lambda v: counts.get(v, 0) / total if total else 1.0)
    weights = prop.apply(lambda p: 1.0 if p <= 0 else min(clip, 1.0 / p))
    return weights.astype(float)
