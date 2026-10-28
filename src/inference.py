from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from data.config import DEFAULT_MODEL_PATH, INFERENCE_NULL_COLUMNS, XGB_PROBABILITY_COL
from data.constants import inferences_table, schema
from load_data import load_holdout_spark_df, save_inferences_spark

REPO_ROOT = Path(__file__).resolve().parents[1]


def infer_holdout(*, model_path: Path) -> None:
    holdout = load_holdout_spark_df()
    for col in INFERENCE_NULL_COLUMNS:
        if col in holdout.columns:
            holdout = holdout.withColumn(col, F.lit(None))

    base_cols = holdout.columns
    pipeline = PipelineModel.load(str(model_path))
    scored = pipeline.transform(holdout)
    scored = scored.withColumn("click_proba", vector_to_array(XGB_PROBABILITY_COL).getItem(1))
    save_inferences_spark(scored.select(*base_cols, "click_proba"))
    print(f"Wrote scored holdout rows to {schema}.{inferences_table}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score holdout week with Spark XGBoost (game_position nullified)")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / DEFAULT_MODEL_PATH)
    args, _ = parser.parse_known_args()

    infer_holdout(model_path=args.model_path)


if __name__ == "__main__":
    main()
