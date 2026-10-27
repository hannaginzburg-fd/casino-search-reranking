from __future__ import annotations

from pathlib import Path

import pandas as pd

from spark_session import get_spark

FINAL_QUERY = """
SELECT *
FROM base_features
WHERE final_similarity_score IS NOT NULL
"""


def _strip_sql_comments(sql: str) -> str:
    lines = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        lines.append(line)
    return "\n".join(lines)


def run_sql_script(sql_path: str | Path) -> None:
    spark = get_spark()
    raw = Path(sql_path).read_text(encoding="utf-8")
    body = _strip_sql_comments(raw)
    for statement in body.split(";"):
        stmt = statement.strip()
        if stmt:
            spark.sql(stmt)


def load_full_df(sql_path: str | Path) -> pd.DataFrame:
    """Execute feature SQL in Spark and return the labeled training frame as pandas."""
    run_sql_script(sql_path)
    spark = get_spark()
    return spark.sql(FINAL_QUERY).toPandas()


def save_full_df(full_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(path, index=False)


def load_full_df_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
