from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from data.constants import (
    contest_history_lookback_days,
    full_df_start_date,
    full_df_table,
    inferences_table,
    schema,
    train_df_table,
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


def load_full_df_table() -> pd.DataFrame:
    return get_spark().table(f"{schema}.{full_df_table}").toPandas()


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
    args = parser.parse_args()

    run_sql_script(args.sql)
    save_full_df()
    row_count = get_spark().table(f"{schema}.{full_df_table}").count()
    print(f"Wrote {row_count:,} rows to {schema}.{full_df_table}")
