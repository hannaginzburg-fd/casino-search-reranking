from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from data.config import EXCLUDE_FROM_TRAINING
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
)
from spark_session import get_spark

FINAL_QUERY = """
SELECT *
FROM base_features
WHERE final_similarity_score IS NOT NULL
"""


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
        .saveAsTable(f"{schema}.{full_df_table}")
    )


def _filtered_train_spark():
    full = get_spark().table(f"{schema}.{full_df_table}")
    max_date = full.agg(F.max("event_date")).first()[0]
    holdout_start = max_date - timedelta(days=holdout_period - 1)
    train = full.filter(F.col("event_date") < F.lit(holdout_start))
    columns = [c for c in train.columns if c not in EXCLUDE_FROM_TRAINING]
    train = train.select(columns)
    if train_sample_fraction < 1.0:
        train = train.sample(withReplacement=False, fraction=train_sample_fraction, seed=42)
    if train_max_rows is not None:
        train = train.orderBy(F.rand(42)).limit(int(train_max_rows))
    return train


def load_train_spark_df():
    train = _filtered_train_spark()
    print(f"Training rows (Spark): {train.count():,}")
    return train


def load_train_df_table() -> pd.DataFrame:
    train = _filtered_train_spark()
    row_count = train.count()
    print(
        f"Collecting {row_count:,} training rows to driver "
        f"(sample_fraction={train_sample_fraction}, max_rows={train_max_rows})"
    )
    return train.toPandas()


def load_holdout_spark_df():
    full = get_spark().table(f"{schema}.{full_df_table}")
    max_date = full.agg(F.max("event_date")).first()[0]
    holdout_start = max_date - timedelta(days=holdout_period - 1)
    holdout = full.filter(F.col("event_date") >= F.lit(holdout_start))
    columns = [c for c in holdout.columns if c != "inserted_ts"]
    return holdout.select(columns)


def load_holdout_df_table() -> pd.DataFrame:
    return load_holdout_spark_df().toPandas()


def save_train_spark_df(train_df) -> None:
    (
        train_df.withColumn("inserted_ts", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .saveAsTable(f"{schema}.{train_df_table}")
    )


def save_inferences_spark(scored_df) -> None:
    (
        scored_df.withColumn("inserted_ts", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .saveAsTable(f"{schema}.{inferences_table}")
    )


def save_train_df(train_df: pd.DataFrame) -> None:
    (
        get_spark()
        .createDataFrame(train_df)
        .withColumn("inserted_ts", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .saveAsTable(f"{schema}.{train_df_table}")
    )


def save_inferences(scored_df: pd.DataFrame) -> None:
    (
        get_spark()
        .createDataFrame(scored_df)
        .withColumn("inserted_ts", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
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
