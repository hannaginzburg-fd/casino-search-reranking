"""Build daily inference full_df Delta (event_date = today) from data_for_inference.sql."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.constants import full_df_for_inference_table, schema
from load_data import run_sql_script, save_full_df_for_inference
from spark_session import get_spark

DEFAULT_SQL = Path(__file__).resolve().parent / "data" / "data_for_inference.sql"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cas_search_reranking_full_df_for_inference for current_date()"
    )
    parser.add_argument("--sql", type=Path, default=DEFAULT_SQL, help="Path to data_for_inference.sql")
    args, _ = parser.parse_known_args()

    run_sql_script(args.sql)
    save_full_df_for_inference()
    row_count = get_spark().table(f"{schema}.{full_df_for_inference_table}").count()
    print(f"Wrote {row_count:,} rows to {schema}.{full_df_for_inference_table}")


if __name__ == "__main__":
    main()
