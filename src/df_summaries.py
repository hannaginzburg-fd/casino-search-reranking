from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import BooleanType, DecimalType, NumericType, StringType

from data.config import LABEL_COLUMN
from features import training_feature_columns
from data.constants import summary_corr_sample_rows

CTR_LABEL = "click_through_rate"
UNIQUE_USERS = "unique_users"

R30_RANK_COLS = [
    "r30_popularity_rank_by_num_users",
    "r30_popularity_rank_by_bet_amt",
    "r30_popularity_rank_by_ggr",
    "r30_popularity_rank_by_ngr",
]

DIMENSION_BAR_CUTS = [
    ("app_id", "platform_app_id", None),
    ("game_super_class", "game_super_class", None),
    ("day_of_week", "day_of_week", None),
    ("day_of_month", "day_of_month", None),
    ("game_position", "game_position", "position"),
    ("recency", "recency", "recency"),
    *[(c, c, "rank") for c in R30_RANK_COLS],
]

CATEGORICAL_OVERRIDES = {
    "app_id",
    "game_super_class",
    "game_class",
    "game_studio_id",
    "day_of_week",
    "day_of_month",
    "month_of_year",
    "week_of_year",
    "first_play_this_year",
}


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def _position_bucket_col(col_name: str = "game_position"):
    c = F.col(col_name)
    return (
        F.when(c.isNull(), F.lit("null"))
        .when(c <= 5, F.lit("1-5"))
        .when(c <= 10, F.lit("6-10"))
        .when(c <= 20, F.lit("11-20"))
        .when(c <= 50, F.lit("21-50"))
        .otherwise(F.lit("51+"))
    )


def _recency_bucket_col(col_name: str = "recency"):
    c = F.col(col_name)
    return (
        F.when(c.isNull(), F.lit("null"))
        .when(c <= 7, F.lit("0-7d"))
        .when(c <= 30, F.lit("8-30d"))
        .when(c <= 90, F.lit("31-90d"))
        .when(c <= 365, F.lit("91-365d"))
        .otherwise(F.lit("366d+"))
    )


def _rank_bucket_col(col_name: str):
    c = F.col(col_name)
    return (
        F.when(c.isNull(), F.lit("null"))
        .when(c <= 10, F.lit("1-10"))
        .when(c <= 50, F.lit("11-50"))
        .when(c <= 100, F.lit("51-100"))
        .when(c <= 500, F.lit("101-500"))
        .otherwise(F.lit("501+"))
    )


def _bucket_expr(source_col: str, bucket_kind: str | None):
    if bucket_kind == "position":
        return _position_bucket_col(source_col)
    if bucket_kind == "recency":
        return _recency_bucket_col(source_col)
    if bucket_kind == "rank":
        return _rank_bucket_col(source_col)
    return F.col(source_col).cast("string")


def _agg_users_ctr(sdf: DataFrame, group_col: str) -> pd.DataFrame:
    pdf = (
        sdf.groupBy(group_col)
        .agg(
            F.countDistinct("fanduel_user_id").alias(UNIQUE_USERS),
            F.avg(LABEL_COLUMN).alias(CTR_LABEL),
            F.count(F.lit(1)).alias("row_count"),
        )
        .orderBy(group_col)
        .toPandas()
    )
    return pdf


def _weekly_metrics(sdf: DataFrame) -> pd.DataFrame:
    return (
        sdf.withColumn("week_start", F.date_trunc("week", F.col("event_date")))
        .groupBy("week_start")
        .agg(
            F.countDistinct("fanduel_user_id").alias(UNIQUE_USERS),
            F.avg(LABEL_COLUMN).alias(CTR_LABEL),
            F.count(F.lit(1)).alias("row_count"),
        )
        .orderBy("week_start")
        .toPandas()
    )


def _line_chart(pdf: pd.DataFrame, x: str, y: str, title: str, path: Path, y_fmt=None) -> None:
    if pdf.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = range(len(pdf))
    ax.plot(xs, pdf[y].values, marker="o", linewidth=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(v)[:10] for v in pdf[x].values], rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(y.replace("_", " "))
    for i, val in enumerate(pdf[y].values):
        label = y_fmt(val) if y_fmt else f"{val:.4f}" if isinstance(val, float) else str(val)
        ax.annotate(label, (i, val), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _bar_chart(pdf: pd.DataFrame, x: str, y: str, title: str, path: Path, y_fmt=None) -> None:
    if pdf.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, min(16, len(pdf) * 0.35)), 4.5))
    bars = ax.bar(pdf[x].astype(str), pdf[y].values, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel(y.replace("_", " "))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    for bar, val in zip(bars, pdf[y].values):
        label = y_fmt(val) if y_fmt else f"{val:.4f}" if isinstance(val, float) else str(val)
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90 if len(pdf) > 15 else 0,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _model_feature_cols(sdf: DataFrame) -> list[str]:
    return training_feature_columns(list(sdf.columns))


def _numeric_and_categorical(sdf: DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for name in feature_cols:
        if name not in sdf.columns:
            continue
        dt = sdf.schema[name].dataType
        if name in CATEGORICAL_OVERRIDES or isinstance(dt, (StringType, BooleanType)):
            categorical.append(name)
        elif isinstance(dt, NumericType):
            numeric.append(name)
    return numeric, categorical


def _quantile_summary(sdf: DataFrame, col: str) -> dict:
    row = sdf.select(
        F.avg(col).alias("avg"),
        F.min(col).alias("min"),
        F.expr(f"percentile_approx(`{col}`, 0.25)").alias("p25"),
        F.expr(f"percentile_approx(`{col}`, 0.5)").alias("median"),
        F.expr(f"percentile_approx(`{col}`, 0.75)").alias("p75"),
        F.max(col).alias("max"),
    ).first()
    if row is None:
        return {}
    stats = row.asDict()
    stats["range"] = (stats["max"] or 0) - (stats["min"] or 0)
    stats["feature"] = col
    return stats


def _ctr_by_quant_bins(sdf: DataFrame, col: str, n_bins: int = 20) -> pd.DataFrame:
    """Quantile bins via approxQuantile (no global Window — safe on large train)."""
    filtered = sdf.filter(F.col(col).isNotNull())
    if filtered.limit(1).count() == 0:
        return pd.DataFrame()
    probs = [i / n_bins for i in range(n_bins + 1)]
    edges = filtered.stat.approxQuantile(col, probs, relativeError=0.05)
    edges = sorted({float(x) for x in edges if x is not None})
    if len(edges) < 2:
        return pd.DataFrame()
    bucket = F.lit(None)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        is_last = i == len(edges) - 2
        in_bin = (F.col(col) >= F.lit(lo)) & (
            F.col(col) <= F.lit(hi) if is_last else F.col(col) < F.lit(hi)
        )
        label = f"{lo:.4g}-{hi:.4g}"
        bucket = F.when(in_bin, F.lit(label)).otherwise(bucket)
    pdf = (
        filtered.withColumn("_bin", bucket)
        .filter(F.col("_bin").isNotNull())
        .groupBy("_bin")
        .agg(F.avg(LABEL_COLUMN).alias(CTR_LABEL), F.count(F.lit(1)).alias("row_count"))
        .orderBy("_bin")
        .toPandas()
        .rename(columns={"_bin": col})
    )
    return pdf


def _write_index(output_dir: Path, paths: list[Path]) -> None:
    lines = ["# Training data summaries\n"]
    for p in sorted(paths):
        lines.append(f"- [{p.name}]({p.name})\n")
    (output_dir / "README.md").write_text("".join(lines), encoding="utf-8")


def run_training_summaries(sdf: DataFrame, *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    if LABEL_COLUMN not in sdf.columns or "fanduel_user_id" not in sdf.columns:
        raise ValueError("Summary frame requires fanduel_user_id and label columns")

    print(f"share_df_summaries: writing charts to {output_dir}")

    weekly = _weekly_metrics(sdf)
    for metric, fmt in ((UNIQUE_USERS, lambda v: f"{int(v):,}"), (CTR_LABEL, lambda v: f"{v:.4f}")):
        fname = output_dir / f"weekly_{metric}.png"
        _line_chart(weekly, "week_start", metric, f"{metric} by calendar week (train)", fname, y_fmt=fmt)
        saved.append(fname)
        print(weekly[["week_start", metric]].to_string(index=False))

    for source_col, slug, bucket_kind in DIMENSION_BAR_CUTS:
        if source_col not in sdf.columns:
            continue
        dim = _bucket_expr(source_col, bucket_kind).alias("dimension")
        cut = _agg_users_ctr(sdf.withColumn("dimension", dim), "dimension")
        for metric, fmt in ((UNIQUE_USERS, lambda v: f"{int(v):,}"), (CTR_LABEL, lambda v: f"{v:.4f}")):
            fname = output_dir / f"bar_{slug}_{metric}.png"
            title = f"{metric} by {slug} (train)"
            _bar_chart(cut, "dimension", metric, title, fname, y_fmt=fmt)
            saved.append(fname)

    feature_cols = _model_feature_cols(sdf)
    numeric_cols, categorical_cols = _numeric_and_categorical(sdf, feature_cols)

    sample = sdf
    total = sdf.count()
    if total > summary_corr_sample_rows:
        frac = min(1.0, summary_corr_sample_rows / total)
        sample = sdf.sample(withReplacement=False, fraction=frac, seed=42)

    if numeric_cols:
        num_sdf = sample.select(*numeric_cols)
        for field in num_sdf.schema.fields:
            if isinstance(field.dataType, DecimalType):
                num_sdf = num_sdf.withColumn(field.name, F.col(field.name).cast("double"))
        num_pdf = num_sdf.toPandas()
    else:
        num_pdf = pd.DataFrame()
    if not num_pdf.empty and len(numeric_cols) > 1:
        corr = num_pdf.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.45), max(6, len(corr) * 0.45)))
        sns.heatmap(corr, annot=len(corr) <= 12, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
        ax.set_title("Feature cross-correlation (numeric, sampled train)")
        fig.tight_layout()
        corr_path = output_dir / "feature_correlation_heatmap.png"
        fig.savefig(corr_path, dpi=120)
        plt.close(fig)
        saved.append(corr_path)

    quant_rows = []
    for col in numeric_cols:
        stats = _quantile_summary(sdf, col)
        if stats:
            quant_rows.append(stats)
        bin_pdf = _ctr_by_quant_bins(sdf, col)
        if not bin_pdf.empty:
            fname = output_dir / f"ctr_by_{_slug(col)}_bins.png"
            _bar_chart(bin_pdf, col, CTR_LABEL, f"CTR by {col} (quantile bins, train)", fname)
            saved.append(fname)
    if quant_rows:
        quant_df = pd.DataFrame(quant_rows)
        quant_df.to_csv(output_dir / "numeric_feature_summaries.csv", index=False)
        print("\nNumeric feature summaries:\n", quant_df.to_string(index=False))

    for col in categorical_cols:
        if col not in sdf.columns:
            continue
        cat = _agg_users_ctr(sdf.withColumn("dimension", F.col(col).cast("string")), "dimension")
        fname = output_dir / f"cat_count_{_slug(col)}.png"
        _bar_chart(cat, "dimension", "row_count", f"Row count by {col} (train)", fname, y_fmt=lambda v: f"{int(v):,}")
        saved.append(fname)
        fname = output_dir / f"cat_ctr_{_slug(col)}.png"
        _bar_chart(cat, "dimension", CTR_LABEL, f"CTR by {col} (train)", fname)
        saved.append(fname)

    _write_index(output_dir, saved)
    print(f"share_df_summaries: wrote {len(saved)} charts (+ README.md, numeric_feature_summaries.csv)")
