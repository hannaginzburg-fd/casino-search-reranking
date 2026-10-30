"""XGBoost feature importance diagnostics (Spark pipeline or driver JSON model)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb
from pyspark.ml import PipelineModel

from data.config import DEFAULT_DRIVER_MODEL_PATH, DEFAULT_FEATURE_LIST_PATH, DEFAULT_SPARK_MODEL_PATH
from data.constants import spark_xgb_training
from features import load_feature_columns

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMPORTANCE_CSV = REPO_ROOT / "artifacts" / "model_feature_importance.csv"


def _display_name(vector_col: str) -> str:
    prefix = "__idx_"
    return vector_col[len(prefix) :] if vector_col.startswith(prefix) else vector_col


def _vector_assembler_input_cols(pipeline: PipelineModel) -> list[str]:
    for stage in pipeline.stages:
        if stage.__class__.__name__ == "VectorAssembler":
            return list(stage.getInputCols())
    raise ValueError("Spark pipeline has no VectorAssembler stage")


def _importance_for_index(
    scores: dict[str, float],
    *,
    index: int,
    vec_col: str,
    display: str,
) -> float:
    for key in (f"f{index}", vec_col, display):
        if key in scores:
            return float(scores[key])
    return 0.0


def _importance_frame(booster: xgb.Booster, vector_feature_names: list[str]) -> pd.DataFrame:
    weight = booster.get_score(importance_type="weight") or {}
    gain = booster.get_score(importance_type="gain") or {}
    cover = booster.get_score(importance_type="cover") or {}

    rows: list[dict] = []
    for i, vec_col in enumerate(vector_feature_names):
        display = _display_name(vec_col)
        split_count = _importance_for_index(weight, index=i, vec_col=vec_col, display=display)
        total_gain = _importance_for_index(gain, index=i, vec_col=vec_col, display=display)
        avg_cover = _importance_for_index(cover, index=i, vec_col=vec_col, display=display)
        rows.append(
            {
                "feature": display,
                "split_count": int(split_count),
                "total_gain": total_gain,
                "avg_gain_per_split": (total_gain / split_count) if split_count else 0.0,
                "avg_cover": avg_cover,
            }
        )

    df = pd.DataFrame(rows)
    total_gain = float(df["total_gain"].sum())
    if total_gain > 0:
        df["gain_share_pct"] = 100.0 * df["total_gain"] / total_gain
    else:
        df["gain_share_pct"] = 0.0
    return df.sort_values(["split_count", "total_gain"], ascending=[False, False]).reset_index(drop=True)


def _load_booster_and_names(
    *,
    model_path: Path,
    feature_list_path: Path,
    use_spark_pipeline: bool,
) -> tuple[xgb.Booster, list[str]]:
    feature_cols = load_feature_columns(feature_list_path)
    if use_spark_pipeline:
        pipeline = PipelineModel.load(str(model_path))
        xgb_stage = pipeline.stages[-1]
        booster = xgb_stage.get_booster()
        vector_names = _vector_assembler_input_cols(pipeline)
        return booster, vector_names

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    if booster.feature_names:
        return booster, list(booster.feature_names)
    return booster, list(feature_cols)


def build_feature_importance_table(
    *,
    model_path: Path,
    feature_list_path: Path,
    use_spark_pipeline: bool | None = None,
) -> pd.DataFrame:
    spark_path = spark_xgb_training if use_spark_pipeline is None else use_spark_pipeline
    booster, vector_names = _load_booster_and_names(
        model_path=model_path,
        feature_list_path=feature_list_path,
        use_spark_pipeline=spark_path,
    )
    return _importance_frame(booster, vector_names)


def print_xgb_model_diagnostics(
    *,
    model_path: Path | None = None,
    feature_list_path: Path | None = None,
    use_spark_pipeline: bool | None = None,
    output_csv: Path | None = DEFAULT_IMPORTANCE_CSV,
) -> pd.DataFrame:
    """
    Print split-based importance from the trained booster.

    - split_count: how often a feature is used to split (XGB ``weight``).
    - total_gain / avg_gain_per_split: loss reduction attributed to that feature (XGB ``gain``).
    - gain_share_pct: share of total gain across features.
    - avg_cover: average coverage on splits using the feature (XGB ``cover``).
    """
    spark_path = spark_xgb_training if use_spark_pipeline is None else use_spark_pipeline
    model_path = model_path or (REPO_ROOT / (DEFAULT_SPARK_MODEL_PATH if spark_path else DEFAULT_DRIVER_MODEL_PATH))
    feature_list_path = feature_list_path or (REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)

    booster, vector_names = _load_booster_and_names(
        model_path=model_path,
        feature_list_path=feature_list_path,
        use_spark_pipeline=spark_path,
    )
    df = _importance_frame(booster, vector_names)
    try:
        n_trees = int(booster.num_boosted_rounds())
    except Exception:
        n_trees = len(booster.get_dump())

    print("=== XGBoost feature importance (trained model) ===")
    print(
        "How to read: split_count = # of tree splits on the feature; "
        "total_gain / gain_share_pct = how much loss improvement those splits contributed; "
        "avg_gain_per_split = total_gain / split_count."
    )
    print(f"Model: {model_path}")
    print(f"Trees in ensemble: {n_trees:,}")
    if df.empty:
        print("No split importance scores returned (empty booster?).")
    else:
        show = df.copy()
        show["total_gain"] = show["total_gain"].map(lambda v: f"{v:.4g}")
        show["avg_gain_per_split"] = show["avg_gain_per_split"].map(lambda v: f"{v:.4g}")
        show["avg_cover"] = show["avg_cover"].map(lambda v: f"{v:.4g}")
        show["gain_share_pct"] = show["gain_share_pct"].map(lambda v: f"{v:.1f}%")
        print(show.to_string(index=False))
        top = df.head(10)
        print("\nTop features by split_count:", ", ".join(top["feature"].tolist()))
        print("Top features by total_gain:", ", ".join(df.sort_values("total_gain", ascending=False).head(10)["feature"].tolist()))

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nWrote full table to {output_csv}")
    print("=" * 48)
    return df
