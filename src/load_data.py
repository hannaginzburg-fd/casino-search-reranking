from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, functions as F

from data.config import (
    EXCLUDE_FROM_TRAINING,
    INVERSE_PROPENSITY_WEIGHTING,
    IPW_POSITION_COLUMN,
    LABEL_COLUMN,
)
from data.constants import sample_weight_col
from ipw import add_inverse_propensity_weights
from data.constants import (
    contest_history_lookback_days,
    full_df_start_date,
    full_df_table,
    holdout_period,
    inferences_table,
    schema,
    train_df_table,
    train_max_rows,
    train_sample_fraction,
    training_lookback_window,
    validation_days,
)
from spark_session import get_spark

FINAL_QUERY = """
SELECT *
FROM base_features
WHERE final_similarity_score IS NOT NULL
"""


def _split_boundaries(max_event_date: date) -> tuple[date, date]:
    """Return (validation_start, holdout_start) for temporal train | validation | holdout."""
    holdout_start = max_event_date - timedelta(days=holdout_period - 1)
    validation_start = holdout_start - timedelta(days=validation_days)
    return validation_start, holdout_start


def _max_event_date(full: DataFrame) -> date:
    return full.agg(F.max("event_date")).first()[0]


def _log_temporal_split(
    *,
    max_event_date: date,
    train: DataFrame,
    validation: DataFrame,
    holdout: DataFrame | None = None,
) -> None:
    validation_start, holdout_start = _split_boundaries(max_event_date)
    train_n = train.count()
    val_n = validation.count()
    msg = (
        f"Temporal split (max event_date={max_event_date}): "
        f"train event_date < {validation_start} ({train_n:,} rows); "
        f"validation [{validation_start}, {holdout_start}) ({val_n:,} rows); "
        f"holdout event_date >= {holdout_start}"
    )
    if holdout is not None:
        msg += f" ({holdout.count():,} rows)"
    print(msg)


def _feature_columns(full: DataFrame) -> list[str]:
    return [c for c in full.columns if c not in EXCLUDE_FROM_TRAINING]


def _train_frame(full: DataFrame, validation_start: date) -> DataFrame:
    cols = [c for c in full.columns if c != "inserted_ts"]
    train = full.filter(F.col("event_date") < F.lit(validation_start)).select(cols)
    if train_sample_fraction < 1.0:
        train = train.sample(withReplacement=False, fraction=train_sample_fraction, seed=42)
    if train_max_rows is not None:
        train = train.orderBy(F.rand(42)).limit(int(train_max_rows))
    if INVERSE_PROPENSITY_WEIGHTING:
        train = add_inverse_propensity_weights(
            train,
            position_col=IPW_POSITION_COLUMN,
            weight_col=sample_weight_col,
        )
    feature_cols = _feature_columns(full)
    out_cols = [*feature_cols, LABEL_COLUMN]
    if INVERSE_PROPENSITY_WEIGHTING:
        out_cols.append(sample_weight_col)
    return train.select(*[c for c in out_cols if c in train.columns])


def _validation_frame(full: DataFrame, validation_start: date, holdout_start: date) -> DataFrame:
    return full.filter(
        (F.col("event_date") >= F.lit(validation_start)) & (F.col("event_date") < F.lit(holdout_start))
    ).select(_feature_columns(full))


def run_sql_script(sql_path: str | Path) -> None:
    spark = get_spark()
    body = Path(sql_path).read_text(encoding="utf-8").format(
        full_df_start_date=full_df_start_date,
        training_lookback_window=training_lookback_window,
        contest_history_lookback_days=contest_history_lookback_days,
    )
    for statement in body.split(";"):
        stmt = statement.strip()
        if stmt:
            spark.sql(stmt)


def save_full_df() -> None:
    (
        get_spark()
        .sql(FINAL_QUERY)
        .withColumn("inserted_ts", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{schema}.{full_df_table}")
    )


def _temporal_train_validation(full: DataFrame) -> tuple[date, DataFrame, DataFrame]:
    max_date = _max_event_date(full)
    validation_start, holdout_start = _split_boundaries(max_date)
    train = _train_frame(full, validation_start)
    validation = _validation_frame(full, validation_start, holdout_start)
    return max_date, train, validation


def load_train_summary_spark_df() -> DataFrame:
    """Full temporal train slice (no feature-column drop, no driver sampling) for EDA."""
    full = get_spark().table(f"{schema}.{full_df_table}")
    max_date = _max_event_date(full)
    validation_start, _ = _split_boundaries(max_date)
    cols = [c for c in full.columns if c != "inserted_ts"]
    return full.filter(F.col("event_date") < F.lit(validation_start)).select(cols)


def load_train_spark_df() -> DataFrame:
    full = get_spark().table(f"{schema}.{full_df_table}")
    max_date, train, validation = _temporal_train_validation(full)
    _log_temporal_split(max_event_date=max_date, train=train, validation=validation)
    print(f"Training rows (Spark): {train.count():,}")
    return train


def load_validation_spark_df() -> DataFrame:
    full = get_spark().table(f"{schema}.{full_df_table}")
    _, train, validation = _temporal_train_validation(full)
    print(f"Validation rows (Spark): {validation.count():,}")
    return validation


def load_train_df_table() -> pd.DataFrame:
    full = get_spark().table(f"{schema}.{full_df_table}")
    max_date, train, validation = _temporal_train_validation(full)
    _log_temporal_split(max_event_date=max_date, train=train, validation=validation)
    row_count = train.count()
    print(
        f"Collecting {row_count:,} training rows to driver "
        f"(sample_fraction={train_sample_fraction}, max_rows={train_max_rows})"
    )
    return train.toPandas()


def load_validation_df_table() -> pd.DataFrame:
    full = get_spark().table(f"{schema}.{full_df_table}")
    _, _, validation = _temporal_train_validation(full)
    row_count = validation.count()
    print(f"Collecting {row_count:,} validation rows to driver")
    return validation.toPandas()


def load_holdout_spark_df() -> DataFrame:
    full = get_spark().table(f"{schema}.{full_df_table}")
    max_date, train, validation = _temporal_train_validation(full)
    _, holdout_start = _split_boundaries(max_date)
    holdout = full.filter(F.col("event_date") >= F.lit(holdout_start))
    _log_temporal_split(max_event_date=max_date, train=train, validation=validation, holdout=holdout)
    columns = [c for c in holdout.columns if c != "inserted_ts"]
    return holdout.select(columns)


def load_holdout_df_table() -> pd.DataFrame:
    return load_holdout_spark_df().toPandas()


def save_train_spark_df(train_df: DataFrame) -> None:
    (
        train_df.withColumn("inserted_ts", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{schema}.{train_df_table}")
    )


def save_validation_spark_df(validation_df: DataFrame) -> None:
    from data.constants import validation_df_table

    (
        validation_df.withColumn("inserted_ts", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{schema}.{validation_df_table}")
    )


def save_inferences_spark(scored_df) -> None:
    (
        scored_df.withColumn("inserted_ts", F.current_timestamp())
        .write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(f"{schema}.{inferences_table}")
    )


def save_train_df(train_df: pd.DataFrame) -> None:
    (
        get_spark()
        .createDataFrame(train_df)
        .withColumn("inserted_ts", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{schema}.{train_df_table}")
    )


def save_inferences(scored_df: pd.DataFrame) -> None:
    (
        get_spark()
        .createDataFrame(scored_df)
        .withColumn("inserted_ts", F.current_timestamp())
        .write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(f"{schema}.{inferences_table}")
    )


if __name__ == "__main__":
    import argparse

    default_sql = Path(__file__).resolve().parent / "data" / "training_data.sql"
    parser = argparse.ArgumentParser(description="Build full_df from feature SQL and write Delta")
    parser.add_argument("--sql", type=Path, default=default_sql, help="Path to training_data.sql")
    args, _ = parser.parse_known_args()

    run_sql_script(args.sql)
    save_full_df()
    row_count = get_spark().table(f"{schema}.{full_df_table}").count()
    print(f"Wrote {row_count:,} rows to {schema}.{full_df_table}")
