"""Build training full_df Delta from data_for_training.sql."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.constants import full_df_for_training_table, schema
from load_data import run_sql_script, save_full_df_for_training
from spark_session import get_spark

DEFAULT_SQL = Path(__file__).resolve().parent / "data" / "data_for_training.sql"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cas_search_reranking_full_df_for_training from feature SQL"
    )
    parser.add_argument("--sql", type=Path, default=DEFAULT_SQL, help="Path to data_for_training.sql")
    args, _ = parser.parse_known_args()

    run_sql_script(args.sql)
    save_full_df_for_training()
    row_count = get_spark().table(f"{schema}.{full_df_for_training_table}").count()
    print(f"Wrote {row_count:,} rows to {schema}.{full_df_for_training_table}")


if __name__ == "__main__":
    main()
