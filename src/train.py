from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import xgboost as xgb
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.types import StringType
from sklearn.metrics import roc_auc_score
from xgboost.spark import SparkXGBClassifier

from data.config import (
    DEFAULT_DRIVER_MODEL_PATH,
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_SPARK_MODEL_PATH,
    DEFAULT_TRAINING_LOSS_PLOT_PATH,
    EXCLUDE_SCORE_OR_RANK_LEAKAGE_FROM_TRAINING,
    INFERENCE_IMPUTE_LEAKAGE_COLS,
    INFERENCE_NULL_COLUMNS,
    INVERSE_PROPENSITY_WEIGHTING,
    IPW_POSITION_COLUMN,
    LABEL_COLUMN,
    SCORE_OR_RANK_LEAKAGE_COLS,
    XGB_PREDICTION_COL,
    XGB_PROBABILITY_COL,
    XGB_VALIDATION_INDICATOR_COL,
    share_df_summaries,
    validate_pipeline_config,
)
from data.constants import sample_weight_col
from artifact_utils import reset_artifacts_dir
from df_summaries import run_training_summaries
from data.constants import (
    driver_model_dbfs_path,
    spark_xgb_training,
    xgb_barrier_mode,
    xgb_auto_scale_pos_weight,
    xgb_early_stopping_rounds,
    xgb_n_estimators,
    xgb_num_workers,
)
from features import (
    assert_frame_has_columns,
    save_feature_columns,
    training_feature_columns,
    training_matrix,
)
from inference_leakage import compute_leakage_imputation_values
from load_data import (
    load_train_df_table,
    load_train_spark_df,
    load_train_summary_spark_df,
    load_validation_df_table,
    load_validation_spark_df,
    print_temporal_split_overview,
    save_train_df,
    save_train_spark_df,
    save_validation_spark_df,
)
from spark_session import get_spark
from spark_xgb_prep import (
    cast_bools_to_double,
    estimate_scale_pos_weight,
    prepare_spark_xgb_fit_union,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_OUTPUT_DIR = REPO_ROOT / "artifacts" / "training_summaries"
LOSS_PLOT_PATH = REPO_ROOT / DEFAULT_TRAINING_LOSS_PLOT_PATH


def _pick_logloss_metric(metrics: dict) -> tuple[str, list[float]] | None:
    for name, values in metrics.items():
        if values and "logloss" in name.lower():
            return name, [float(v) for v in values]
    if metrics:
        name, values = next(iter(metrics.items()))
        if values:
            return name, [float(v) for v in values]
    return None


def _logloss_series_from_spark_xgb_stage(xgb_stage) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    summary = getattr(xgb_stage, "training_summary", None)
    if summary is not None:
        picked = _pick_logloss_metric(summary.validation_objective_history or {})
        if picked:
            metric, values = picked
            series[f"validation ({metric})"] = values
        picked = _pick_logloss_metric(summary.train_objective_history or {})
        if picked:
            metric, values = picked
            series[f"train ({metric})"] = values
    return series


def _safe_evals_result(model) -> dict:
    if not hasattr(model, "evals_result"):
        return {}
    try:
        evals = model.evals_result()
        if callable(evals):
            evals = evals()
        return evals if isinstance(evals, dict) else {}
    except Exception as exc:
        if type(exc).__name__ == "XGBoostError":
            return {}
        raise


def _logloss_series_from_driver_model(model: xgb.XGBClassifier) -> dict[str, list[float]]:
    evals = _safe_evals_result(model)
    series: dict[str, list[float]] = {}
    for dataset, metrics in evals.items():
        picked = _pick_logloss_metric(metrics)
        if picked:
            metric, values = picked
            series[f"{dataset} ({metric})"] = values
    return series


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


def _save_logloss_plot(series: dict[str, list[float]], output_path: Path) -> bool:
    if not series:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, values in series.items():
        if not values:
            continue
        rounds = list(range(1, len(values) + 1))
        ax.plot(rounds, values, label=label, linewidth=1.5)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Log loss")
    ax.set_title("Training log loss by round")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    _display_figure(fig)
    plt.close(fig)
    return True


def print_position_bias_strategy() -> None:
    if INVERSE_PROPENSITY_WEIGHTING:
        print(
            "Position bias strategy: IPW (inverse propensity weighting on train; "
            f"{IPW_POSITION_COLUMN!r} excluded from model features)"
        )
        return
    if EXCLUDE_SCORE_OR_RANK_LEAKAGE_FROM_TRAINING:
        print(
            "Position bias strategy: leakage columns excluded from training features "
            f"({SCORE_OR_RANK_LEAKAGE_COLS}); no inference null/impute in this mode"
        )
        return
    if INFERENCE_IMPUTE_LEAKAGE_COLS:
        print("Position bias strategy: mean/mode imputation on leakage columns at inference")
        train_slice = load_train_summary_spark_df()
        feature_cols = training_feature_columns(list(train_slice.columns))
        imputation = compute_leakage_imputation_values(train_slice, feature_cols)
        for col in INFERENCE_NULL_COLUMNS:
            if col not in feature_cols:
                continue
            value = imputation.get(col)
            print(f"  {col} imputation (temporal train slice, same rule as inference batch): {value!r}")
        return
    print("Position bias strategy: nullify leakage columns on scoring copy at inference")
    for col in INFERENCE_NULL_COLUMNS:
        print(f"  {col} -> null on model input (original logged in holdout / Delta output)")


def print_training_run_header() -> None:
    print("=== Training run ===")
    print_temporal_split_overview()
    print_position_bias_strategy()
    print("====================")


def _spark_task_slots() -> int:
    spark = get_spark()
    conf = spark.sparkContext.getConf()
    instances = conf.get("spark.executor.instances")
    cores = int(conf.get("spark.executor.cores", "1") or "1")
    if instances and str(instances).isdigit():
        return max(1, int(instances) * cores)
    # Dynamic allocation / Databricks: use default parallelism as a conservative slot estimate
    return max(1, spark.sparkContext.defaultParallelism)


def _resolve_xgb_num_workers(train_sdf, *, requested: int | None) -> int:
    want = requested if requested is not None else 8
    slots = _spark_task_slots()
    parts = max(1, train_sdf.rdd.getNumPartitions())
    effective = max(1, min(want, slots, parts))
    if effective != want:
        print(
            f"Adjusted xgb num_workers {want} -> {effective} "
            f"(task_slots~{slots}, train_partitions={parts})"
        )
    if xgb_barrier_mode and effective < 2:
        print(
            "Warning: barrier_mode with num_workers=1 is fragile; "
            "set xgb_barrier_mode=False or add executors."
        )
    return effective


def _maybe_share_df_summaries() -> None:
    if not share_df_summaries:
        return
    run_training_summaries(load_train_summary_spark_df(), output_dir=SUMMARY_OUTPUT_DIR)


def _prep_spark_features(sdf, feature_cols):
    assert_frame_has_columns(list(sdf.columns), feature_cols, frame_name="Spark XGB prep")
    sdf = cast_bools_to_double(sdf)
    prep_stages = []
    vec_cols = []
    for col in feature_cols:
        if isinstance(sdf.schema[col].dataType, StringType):
            indexed = f"__idx_{col}"
            prep_stages.append(StringIndexer(inputCol=col, outputCol=indexed, handleInvalid="keep"))
            vec_cols.append(indexed)
        else:
            vec_cols.append(col)
    prep_stages.append(VectorAssembler(inputCols=vec_cols, outputCol="features", handleInvalid="keep"))
    return sdf, prep_stages


def train_spark(*, model_path: Path, feature_list_path: Path, random_state: int = 42) -> dict[str, float]:
    train_sdf = load_train_spark_df()
    val_sdf = load_validation_spark_df()
    feature_cols = training_feature_columns(list(train_sdf.columns))
    assert_frame_has_columns(list(train_sdf.columns), feature_cols, frame_name="Train")
    assert_frame_has_columns(
        list(val_sdf.columns),
        [*feature_cols, LABEL_COLUMN],
        frame_name="Validation",
    )
    if not feature_cols:
        raise ValueError("No training features found after exclusions; check full_df schema and config")
    if INVERSE_PROPENSITY_WEIGHTING and sample_weight_col not in train_sdf.columns:
        raise ValueError(
            f"IPW enabled but {sample_weight_col!r} missing from train frame "
            f"(check {IPW_POSITION_COLUMN!r} on full_df and reload train slice)"
        )
    if INVERSE_PROPENSITY_WEIGHTING:
        print(f"Inverse propensity weighting on train ({IPW_POSITION_COLUMN} marginal); validation unweighted")

    save_feature_columns(feature_list_path, feature_cols)
    model_cols = list(dict.fromkeys([*feature_cols, LABEL_COLUMN]))
    save_train_spark_df(train_sdf.select(*model_cols))
    save_validation_spark_df(val_sdf.select(*model_cols))

    train_sdf, val_sdf, fit_sdf = prepare_spark_xgb_fit_union(train_sdf, val_sdf, feature_cols)
    _, prep_stages = _prep_spark_features(train_sdf, feature_cols)

    num_workers = _resolve_xgb_num_workers(fit_sdf, requested=xgb_num_workers)
    train_rows = train_sdf.count()
    val_rows = val_sdf.count()
    if val_rows == 0:
        raise ValueError("Validation slice has 0 rows; rebuild full_df or check temporal split constants")
    scale_pos_weight = None
    if xgb_auto_scale_pos_weight:
        scale_pos_weight = estimate_scale_pos_weight(train_sdf)
        if scale_pos_weight is not None:
            print(f"Spark XGB scale_pos_weight={scale_pos_weight:.4f} (neg/pos on train)")
    print(
        f"SparkXGBClassifier num_workers={num_workers}, barrier_mode={xgb_barrier_mode}, "
        f"n_estimators<={xgb_n_estimators}, early_stopping_rounds={xgb_early_stopping_rounds} "
        f"(train rows ~{train_rows:,}, validation rows ~{val_rows:,}) "
        f"— use on-demand executors; spot preemption aborts barrier training"
    )

    xgb_kwargs: dict = {
        "features_col": "features",
        "label_col": LABEL_COLUMN,
        "prediction_col": XGB_PREDICTION_COL,
        "probability_col": XGB_PROBABILITY_COL,
        "validation_indicator_col": XGB_VALIDATION_INDICATOR_COL,
        "num_workers": num_workers,
        "force_repartition": True,
        "use_gpu": False,
        "barrier_mode": xgb_barrier_mode,
        "eval_metric": "logloss",
        "random_state": random_state,
        "verbosity": 0,
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": xgb_n_estimators,
        "early_stopping_rounds": xgb_early_stopping_rounds,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    if scale_pos_weight is not None:
        xgb_kwargs["scale_pos_weight"] = scale_pos_weight
    if INVERSE_PROPENSITY_WEIGHTING:
        xgb_kwargs["weight_col"] = sample_weight_col
    xgb_estimator = SparkXGBClassifier(**xgb_kwargs)

    pipeline = Pipeline(stages=prep_stages + [xgb_estimator])
    fit_sdf.cache()
    model = pipeline.fit(fit_sdf)
    fit_sdf.unpersist()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str(model_path))

    val_pred = model.transform(val_sdf)
    val_auc = BinaryClassificationEvaluator(
        labelCol=LABEL_COLUMN,
        rawPredictionCol=XGB_PROBABILITY_COL,
        metricName="areaUnderROC",
    ).evaluate(val_pred)

    xgb_stage = model.stages[-1]
    loss_series = _logloss_series_from_spark_xgb_stage(xgb_stage)
    if _save_logloss_plot(loss_series, LOSS_PLOT_PATH):
        print(f"Saved and displayed training log-loss curve at {LOSS_PLOT_PATH}")
    else:
        print("No per-round log-loss history on Spark model; skipping loss plot")

    return {"val_auc": float(val_auc), "train_rows": train_rows, "val_rows": val_rows}


def train_driver(*, model_path: Path, feature_list_path: Path, random_state: int = 42) -> dict[str, float]:
    train_pdf = load_train_df_table()
    val_pdf = load_validation_df_table()
    X_train, y_train, feature_cols = training_matrix(train_pdf)
    X_val, y_val, val_feature_cols = training_matrix(val_pdf)
    if val_feature_cols != feature_cols:
        raise ValueError(
            "Train/validation feature mismatch: "
            f"train-only={set(feature_cols) - set(val_feature_cols)!r}, "
            f"val-only={set(val_feature_cols) - set(feature_cols)!r}"
        )
    save_feature_columns(feature_list_path, feature_cols)
    train_out = X_train.copy()
    train_out[LABEL_COLUMN] = y_train.to_numpy()
    save_train_df(train_out)
    val_out = X_val.copy()
    val_out[LABEL_COLUMN] = y_val.to_numpy()
    save_validation_spark_df(get_spark().createDataFrame(val_out))

    for frame in (X_train, X_val):
        for col in frame.columns:
            if frame[col].dtype == "object" or str(frame[col].dtype) == "string":
                frame[col] = frame[col].astype("category")

    driver_xgb_kwargs: dict = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": xgb_n_estimators,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "enable_categorical": True,
        "tree_method": "hist",
    }
    if xgb_auto_scale_pos_weight:
        pos = int(y_train.sum())
        neg = len(y_train) - pos
        if pos > 0 and neg > 0:
            driver_xgb_kwargs["scale_pos_weight"] = neg / pos
            print(f"Driver XGB scale_pos_weight={driver_xgb_kwargs['scale_pos_weight']:.4f}")
    model = xgb.XGBClassifier(**driver_xgb_kwargs)
    train_weight = None
    if INVERSE_PROPENSITY_WEIGHTING:
        if sample_weight_col not in train_pdf.columns:
            raise ValueError(f"IPW enabled but {sample_weight_col!r} missing from train frame")
        train_weight = train_pdf[sample_weight_col].astype(float).values
        print(f"Inverse propensity weighting on train ({IPW_POSITION_COLUMN} marginal); validation unweighted")
    model.fit(
        X_train,
        y_train,
        sample_weight=train_weight,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=xgb_early_stopping_rounds,
        verbose=False,
    )
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    dbfs_path = Path(driver_model_dbfs_path)
    if dbfs_path.parent.exists():
        dbfs_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_path, dbfs_path)
        print(f"Copied driver model to {driver_model_dbfs_path} for distributed inference")
    loss_series = _logloss_series_from_driver_model(model)
    if _save_logloss_plot(loss_series, LOSS_PLOT_PATH):
        print(f"Saved and displayed training log-loss curve at {LOSS_PLOT_PATH}")
    else:
        print("No eval log-loss history on driver model; skipping loss plot")
    return {"val_auc": float(val_auc), "train_rows": len(X_train), "val_rows": len(X_val)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost from full_df Delta (temporal validation + holdout excluded from fit)"
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--feature-list-path", type=Path, default=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH)
    args, _ = parser.parse_known_args()

    validate_pipeline_config()
    get_spark()
    reset_artifacts_dir()
    print(f"Cleared prior outputs under {REPO_ROOT / 'artifacts'}")
    print_training_run_header()
    _maybe_share_df_summaries()
    if spark_xgb_training:
        model_path = args.model_path or (REPO_ROOT / DEFAULT_SPARK_MODEL_PATH)
        metrics = train_spark(model_path=model_path, feature_list_path=args.feature_list_path)
    else:
        model_path = args.model_path or (REPO_ROOT / DEFAULT_DRIVER_MODEL_PATH)
        metrics = train_driver(model_path=model_path, feature_list_path=args.feature_list_path)

    print(f"Training complete: {metrics}")


if __name__ == "__main__":
    main()
