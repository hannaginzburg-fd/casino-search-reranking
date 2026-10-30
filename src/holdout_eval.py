"""Holdout ranking / calibration diagnostics (data not used for model fit)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StructField,
    StructType,
)
from pyspark.sql.window import Window

from data.config import LABEL_COLUMN, model_position_debias_strategy
from data.constants import inference_eval_metrics_table, schema
from load_data import save_holdout_ranking_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_PLOT = REPO_ROOT / "artifacts" / "holdout_calibration.png"

SEARCH_KEYS = ("fanduel_user_id", "session_id", "search_keyword")
NDCG_K_VALUES = (1, 5, 10)
PRECISION_RECALL_K = (1, 5, 10)


def _dcg_binary(relevances: np.ndarray) -> float:
    if relevances.size == 0:
        return 0.0
    rel = relevances.astype(float)
    positions = np.arange(1, rel.size + 1)
    return float(np.sum(rel / np.log2(positions + 1)))


def _ndcg_binary(relevances_ordered: np.ndarray) -> float:
    rel = relevances_ordered.astype(float)
    dcg = _dcg_binary(rel)
    ideal = _dcg_binary(np.sort(rel)[::-1])
    return dcg / ideal if ideal > 0 else 0.0


def _precision_recall_at_k(relevances_ordered: np.ndarray, k: int) -> tuple[float, float]:
    rel = relevances_ordered.astype(float)
    if rel.size == 0:
        return 0.0, 0.0
    k = min(k, rel.size)
    top = rel[:k]
    clicks = float(rel.sum())
    precision = float(top.sum()) / k
    recall = float(top.sum()) / clicks if clicks > 0 else 0.0
    return precision, recall


def _search_group_metrics(pdf: pd.DataFrame) -> pd.DataFrame:
    if pdf.empty:
        return pd.DataFrame(
            columns=[
                "n_games",
                "n_clicks",
                *[f"ndcg{k}_baseline" for k in NDCG_K_VALUES],
                *[f"ndcg{k}_rerank" for k in NDCG_K_VALUES],
                *[f"precision{k}_baseline" for k in PRECISION_RECALL_K],
                *[f"recall{k}_baseline" for k in PRECISION_RECALL_K],
                *[f"precision{k}_rerank" for k in PRECISION_RECALL_K],
                *[f"recall{k}_rerank" for k in PRECISION_RECALL_K],
            ]
        )
    baseline = pdf.sort_values("game_position", ascending=True, na_position="last")
    rerank = pdf.sort_values("click_proba", ascending=False)
    rel_b = baseline[LABEL_COLUMN].to_numpy()
    rel_r = rerank[LABEL_COLUMN].to_numpy()
    row: dict = {
        "n_games": len(pdf),
        "n_clicks": int(pdf[LABEL_COLUMN].sum()),
    }
    for k in NDCG_K_VALUES:
        k_b = min(k, len(rel_b))
        k_r = min(k, len(rel_r))
        row[f"ndcg{k}_baseline"] = _ndcg_binary(rel_b[:k_b])
        row[f"ndcg{k}_rerank"] = _ndcg_binary(rel_r[:k_r])
    for k in PRECISION_RECALL_K:
        p_b, r_b = _precision_recall_at_k(rel_b, k)
        p_r, r_r = _precision_recall_at_k(rel_r, k)
        row[f"precision{k}_baseline"] = p_b
        row[f"recall{k}_baseline"] = r_b
        row[f"precision{k}_rerank"] = p_r
        row[f"recall{k}_rerank"] = r_r
    return pd.DataFrame([row])


_GROUP_METRICS_SCHEMA = StructType(
    [
        StructField("n_games", IntegerType(), False),
        StructField("n_clicks", LongType(), False),
        *[StructField(f"ndcg{k}_baseline", DoubleType(), False) for k in NDCG_K_VALUES],
        *[StructField(f"ndcg{k}_rerank", DoubleType(), False) for k in NDCG_K_VALUES],
        *[StructField(f"precision{k}_baseline", DoubleType(), False) for k in PRECISION_RECALL_K],
        *[StructField(f"recall{k}_baseline", DoubleType(), False) for k in PRECISION_RECALL_K],
        *[StructField(f"precision{k}_rerank", DoubleType(), False) for k in PRECISION_RECALL_K],
        *[StructField(f"recall{k}_rerank", DoubleType(), False) for k in PRECISION_RECALL_K],
    ]
)


def _add_reranked_position(scored: DataFrame) -> DataFrame:
    window = Window.partitionBy(*SEARCH_KEYS).orderBy(F.col("click_proba").desc())
    return scored.withColumn("reranked_game_position", F.row_number().over(window))


def _display_figure(fig) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(fig)
            return
    except ImportError:
        pass
    plt.show()


def _holdout_eval_context(ranked: DataFrame) -> dict:
    if "event_date" not in ranked.columns:
        raise ValueError("Holdout diagnostics require event_date on the scored holdout frame")
    row = ranked.agg(
        F.min("event_date").alias("min_date"),
        F.max("event_date").alias("max_date"),
        F.countDistinct("fanduel_user_id").alias("num_unique_users"),
        F.countDistinct("game_id").alias("num_unique_games"),
        F.sum(LABEL_COLUMN).alias("total_clicks"),
        F.count(F.lit(1)).alias("total_views"),
    ).first()
    search_counts = (
        ranked.groupBy(*SEARCH_KEYS)
        .agg(F.max(LABEL_COLUMN).alias("_any_click"))
        .agg(
            F.count(F.lit(1)).alias("total_searches_viewed"),
            F.sum(F.when(F.col("_any_click") == 1, 1).otherwise(0)).alias("total_searches_with_click"),
        )
        .first()
    )
    return {
        "min_date": row["min_date"],
        "max_date": row["max_date"],
        "model_position_debias_strategy": model_position_debias_strategy(),
        "num_unique_games": int(row["num_unique_games"] or 0),
        "num_unique_users": int(row["num_unique_users"] or 0),
        "total_clicks": int(row["total_clicks"] or 0),
        "total_views": int(row["total_views"] or 0),
        "total_searches_with_click": int(search_counts["total_searches_with_click"] or 0),
        "total_searches_viewed": int(search_counts["total_searches_viewed"] or 0),
    }


def _metric_row(
    metric: str,
    baseline: float | None,
    reranked: float | None,
    *,
    context: dict,
) -> dict:
    return {
        "metric": metric,
        "baseline": baseline,
        "reranked": reranked,
        "min_date": context["min_date"],
        "max_date": context["max_date"],
        "model_position_debias_strategy": context["model_position_debias_strategy"],
        "num_unique_games": context["num_unique_games"],
        "num_unique_users": context["num_unique_users"],
        "total_clicks": context["total_clicks"],
        "total_views": context["total_views"],
        "total_searches_with_click": context["total_searches_with_click"],
        "total_searches_viewed": context["total_searches_viewed"],
    }


def _ranking_metrics_rows(
    summary,
    *,
    avg_clicked_baseline: float | None,
    avg_clicked_reranked: float | None,
    context: dict,
) -> list[dict]:
    rows = [
        _metric_row("avg_clicked_game_position", avg_clicked_baseline, avg_clicked_reranked, context=context),
    ]
    for k in NDCG_K_VALUES:
        rows.append(
            _metric_row(
                f"ndcg@{k}",
                summary[f"ndcg{k}_baseline"],
                summary[f"ndcg{k}_rerank"],
                context=context,
            )
        )
    for k in PRECISION_RECALL_K:
        rows.append(
            _metric_row(
                f"precision@{k}",
                summary[f"precision{k}_baseline"],
                summary[f"precision{k}_rerank"],
                context=context,
            )
        )
        rows.append(
            _metric_row(
                f"recall@{k}",
                summary[f"recall{k}_baseline"],
                summary[f"recall{k}_rerank"],
                context=context,
            )
        )
    return rows


def _plot_calibration(bucket_pdf: pd.DataFrame, path: Path) -> None:
    if bucket_pdf.empty:
        return
    pdf = bucket_pdf.sort_values("bucket_mid")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration (y=x)")
    ax.plot(pdf["bucket_mid"], pdf["actual_ctr"], marker="o", label="Holdout actual CTR")
    ax.set_xlabel("Predicted click probability (bucket midpoint)")
    ax.set_ylabel("Actual CTR (sum(label) / count)")
    ax.set_title("Holdout calibration by click_proba (0.05-wide buckets)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, min(1.0, max(0.05, pdf["actual_ctr"].max() * 1.15)))
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    _display_figure(fig)
    plt.close(fig)


def print_holdout_ranking_diagnostics(
    scored: DataFrame,
    *,
    calibration_plot_path: Path | None = DEFAULT_CALIBRATION_PLOT,
) -> None:
    """
    Evaluate ranking on holdout (labels + click_proba).

    - Avg slot among clicks: logged ``game_position`` vs model ``reranked_game_position``.
    - Calibration curve vs y=x in 0.05 ``click_proba`` buckets.
    - Macro-averaged NDCG@K and precision/recall@K: baseline order (``game_position``) vs rerank (``click_proba``).
    """
    required = {*SEARCH_KEYS, "game_position", LABEL_COLUMN, "click_proba"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"Holdout diagnostics missing columns: {sorted(missing)}")

    ranked = _add_reranked_position(
        scored.filter(F.col("click_proba").isNotNull() & F.col(LABEL_COLUMN).isNotNull())
    )
    n_rows = ranked.count()
    context = _holdout_eval_context(ranked)
    print("=== Holdout ranking diagnostics (not used in training) ===")
    print(f"Scored holdout rows: {n_rows:,}")
    print(
        f"Holdout window: {context['min_date']} .. {context['max_date']} | "
        f"users={context['num_unique_users']:,} games={context['num_unique_games']:,} | "
        f"views={context['total_views']:,} clicks={context['total_clicks']:,} | "
        f"searches={context['total_searches_viewed']:,} "
        f"searches_with_click={context['total_searches_with_click']:,} | "
        f"position debias strategy={context['model_position_debias_strategy']}"
    )

    clicks = ranked.filter(F.col(LABEL_COLUMN) == 1)
    click_n = clicks.count()
    avg_clicked_baseline = None
    avg_clicked_reranked = None
    if click_n == 0:
        print("No clicked rows (label=1) on holdout; skipping avg clicked slot metrics.")
    else:
        pos = clicks.agg(
            F.avg("game_position").alias("avg_clicked_game_position_baseline"),
            F.avg("reranked_game_position").alias("avg_clicked_game_position_reranked"),
        ).first()
        avg_clicked_baseline = float(pos["avg_clicked_game_position_baseline"])
        avg_clicked_reranked = float(pos["avg_clicked_game_position_reranked"])
        print(
            f"Avg slot among clicks — baseline (logged game_position): "
            f"{avg_clicked_baseline:.4f}"
        )
        print(
            f"Avg slot among clicks — reranked (by click_proba desc): "
            f"{avg_clicked_reranked:.4f}"
        )

    bucket_lo = (F.floor(F.col("click_proba") / F.lit(0.05)) * F.lit(0.05)).alias("bucket_lo")
    cal = (
        ranked.withColumn("bucket_lo", bucket_lo)
        .groupBy("bucket_lo")
        .agg(
            F.count(F.lit(1)).alias("n"),
            F.sum(LABEL_COLUMN).alias("click_sum"),
            F.avg("click_proba").alias("mean_click_proba"),
        )
        .orderBy("bucket_lo")
        .toPandas()
    )
    if not cal.empty:
        cal["actual_ctr"] = cal["click_sum"] / cal["n"]
        cal["bucket_mid"] = cal["bucket_lo"] + 0.025
        print("\nCalibration buckets (0.05 width):")
        show = cal[["bucket_lo", "n", "actual_ctr", "mean_click_proba"]].copy()
        show["actual_ctr"] = show["actual_ctr"].map(lambda v: f"{v:.4f}")
        show["mean_click_proba"] = show["mean_click_proba"].map(lambda v: f"{v:.4f}")
        print(show.to_string(index=False))
        if calibration_plot_path is not None:
            _plot_calibration(cal, calibration_plot_path)
            print(f"Saved calibration plot to {calibration_plot_path}")

    group_cols = list(SEARCH_KEYS)
    metrics = ranked.groupBy(*group_cols).applyInPandas(_search_group_metrics, schema=_GROUP_METRICS_SCHEMA)
    summary = metrics.agg(
        F.count(F.lit(1)).alias("n_search_contexts"),
        F.sum("n_clicks").alias("total_clicks"),
        *[F.avg(f"ndcg{k}_baseline").alias(f"ndcg{k}_baseline") for k in NDCG_K_VALUES],
        *[F.avg(f"ndcg{k}_rerank").alias(f"ndcg{k}_rerank") for k in NDCG_K_VALUES],
        *[F.avg(f"precision{k}_baseline").alias(f"precision{k}_baseline") for k in PRECISION_RECALL_K],
        *[F.avg(f"recall{k}_baseline").alias(f"recall{k}_baseline") for k in PRECISION_RECALL_K],
        *[F.avg(f"precision{k}_rerank").alias(f"precision{k}_rerank") for k in PRECISION_RECALL_K],
        *[F.avg(f"recall{k}_rerank").alias(f"recall{k}_rerank") for k in PRECISION_RECALL_K],
    ).first()

    print(f"\nSearch contexts (user / session / keyword): {summary['n_search_contexts']:,}")
    print(f"Total clicks in holdout: {int(summary['total_clicks'] or 0):,}")
    for k in NDCG_K_VALUES:
        print(
            f"Macro-avg NDCG@{k} — baseline (game_position): {summary[f'ndcg{k}_baseline']:.4f}, "
            f"rerank (click_proba): {summary[f'ndcg{k}_rerank']:.4f}"
        )
    for k in PRECISION_RECALL_K:
        print(
            f"Macro-avg precision@{k} — baseline: {summary[f'precision{k}_baseline']:.4f}, "
            f"rerank: {summary[f'precision{k}_rerank']:.4f}"
        )
        print(
            f"Macro-avg recall@{k}    — baseline: {summary[f'recall{k}_baseline']:.4f}, "
            f"rerank: {summary[f'recall{k}_rerank']:.4f}"
        )

    metric_rows = _ranking_metrics_rows(
        summary,
        avg_clicked_baseline=avg_clicked_baseline,
        avg_clicked_reranked=avg_clicked_reranked,
        context=context,
    )
    save_holdout_ranking_metrics(metric_rows)
    print(
        f"Appended {len(metric_rows)} holdout ranking metric rows to "
        f"{schema}.{inference_eval_metrics_table} (inserted_ts on each row)"
    )
    print("=" * 48)
