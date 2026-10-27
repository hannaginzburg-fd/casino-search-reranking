from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from data.constants import (
    contest_history_lookback_days,
    full_df_start_date,
    full_df_table,
    inferences_table,
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


def load_full_df(sql_path: str | Path) -> pd.DataFrame:
    """Execute feature SQL in Spark and return the labeled training frame as pandas."""
    run_sql_script(sql_path)
    spark = get_spark()
    return spark.sql(FINAL_QUERY).toPandas()


def save_full_df(full_df: pd.DataFrame) -> None:
    (
        get_spark()
        .createDataFrame(full_df)
        .withColumn("inserted_ts", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .saveAsTable(full_df_table)
    )


def load_full_df_table() -> pd.DataFrame:
    return get_spark().table(full_df_table).toPandas()


def save_train_df(train_df: pd.DataFrame) -> None:
    (
        get_spark()
        .createDataFrame(train_df)
        .withColumn("inserted_ts", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .saveAsTable(train_df_table)
    )


def save_inferences(scored_df: pd.DataFrame) -> None:
    (
        get_spark()
        .createDataFrame(scored_df)
        .withColumn("inserted_ts", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .saveAsTable(inferences_table)
    )
