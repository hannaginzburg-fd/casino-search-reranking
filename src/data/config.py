"""Pipeline config — XGB hyperparameters at top; feature & column rules below."""

# ---------------------------------------------------------------------------
# XGBoost — shared tree hyperparameters (SparkXGBClassifier + XGBClassifier)
# ---------------------------------------------------------------------------
xgb_objective = "binary:logistic"
xgb_eval_metric = "logloss"
xgb_max_depth = 8
xgb_learning_rate = 0.05
xgb_n_estimators = 500
xgb_early_stopping_rounds = 30  # temporal validation; Spark + driver fit()
xgb_subsample = 0.8
xgb_colsample_bytree = 0.8
xgb_min_child_weight = 1
xgb_reg_alpha = 0.0
xgb_reg_lambda = 1.0
xgb_train_random_state = 42
xgb_auto_scale_pos_weight = True  # set scale_pos_weight = neg/pos on train click rate

# ---------------------------------------------------------------------------
# XGBoost — driver-only (xgboost.XGBClassifier)
# ---------------------------------------------------------------------------
xgb_tree_method = "hist"
xgb_enable_categorical = True
xgb_driver_fit_verbose = False

# ---------------------------------------------------------------------------
# XGBoost — Spark-only (SparkXGBClassifier)
# ---------------------------------------------------------------------------
xgb_num_workers = None  # None = auto-cap to cluster task slots at train time
xgb_barrier_mode = True
xgb_spark_verbosity = 0
xgb_spark_use_gpu = False
xgb_spark_force_repartition = True

# ---------------------------------------------------------------------------
# IPW (position-bias row weights on train loss)
# ---------------------------------------------------------------------------
ipw_weight_clip_max = 100.0

# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------
DEFAULT_SPARK_MODEL_PATH = "artifacts/xgb_spark_pipeline"
DEFAULT_DRIVER_MODEL_PATH = "artifacts/xgb_model.json"
DEFAULT_FEATURE_LIST_PATH = "artifacts/feature_columns.json"
DEFAULT_TRAINING_LOSS_PLOT_PATH = "artifacts/training_logloss.png"
driver_model_dbfs_path = "/dbfs/FileStore/cas_search_reranking/xgb_model.json"

# Spark ML / XGB stage column names
XGB_PREDICTION_COL = "click_pred"
XGB_PROBABILITY_COL = "click_proba_vector"
XGB_VALIDATION_INDICATOR_COL = "_is_validation"

# ---------------------------------------------------------------------------
# Data pipeline constants (tables, date windows, …)
# ---------------------------------------------------------------------------
from data.constants import *  # noqa: F403

LABEL_COLUMN = "label"

##### 3 Options for Handling Position Bias: #####
# 1) Nullifying game position at inference time
# 2) Mean-imputing game position at inference time
# 3) IPW weighting - inversely weight each labeled row in *training*; inference unchanged

# Position-bias IPW: drop position from features, row-weight train loss; no inference leakage handling.
INVERSE_PROPENSITY_WEIGHTING = True
IPW_POSITION_COLUMN = "game_position"

# Before model fit: EDA charts under artifacts/training_summaries/<split>/ (Spark aggregations; no driver sampling).
share_df_summaries = False
# Splits to summarize when share_df_summaries is True: "train", "validation", "holdout" (any subset, in order).
#share_df_summaries_splits = ["train"]
share_df_summaries_splits = ["train", "validation", "holdout"]

# Logged slot + baseline search score — same policy for train and serve (see below).
SCORE_OR_RANK_LEAKAGE_COLS = ["game_position"]

# When True, neither column is included as a model feature at all (vs. leakage handling at inference).
EXCLUDE_SCORE_OR_RANK_LEAKAGE_FROM_TRAINING = False

# Kept in full_df for analysis / joins; dropped before model fit.
_EXCLUDE_BASE = [
    "fanduel_user_id",
    "session_id",
    "game_id",
    "event_date",
    "search_keyword",
    #"base_similarity_score",
    "final_similarity_score",
    "inserted_ts",
    sample_weight_col,
]


def training_feature_exclude() -> list[str]:
    exclude = list(_EXCLUDE_BASE)
    if EXCLUDE_SCORE_OR_RANK_LEAKAGE_FROM_TRAINING:
        for col in SCORE_OR_RANK_LEAKAGE_COLS:
            if col not in exclude:
                exclude.append(col)
    if INVERSE_PROPENSITY_WEIGHTING:
        if IPW_POSITION_COLUMN not in exclude:
            exclude.append(IPW_POSITION_COLUMN)
    return exclude


# Back-compat name used across loaders / features.
EXCLUDE_FROM_TRAINING = training_feature_exclude()

# Leakage columns adjusted on the scoring copy only (mutually exclusive with IPW).
INFERENCE_NULL_COLUMNS = list(SCORE_OR_RANK_LEAKAGE_COLS)

# True: mean/mode imputation on scoring copy. Must be False when INVERSE_PROPENSITY_WEIGHTING is True.
INFERENCE_IMPUTE_LEAKAGE_COLS = False


def model_position_debias_strategy() -> str:
    """Label for inference_eval_metrics Delta (matches train/inference config)."""
    if INVERSE_PROPENSITY_WEIGHTING:
        return "ipw"
    if INFERENCE_IMPUTE_LEAKAGE_COLS:
        return "mean_imputed"
    return "nullify"


def validate_pipeline_config() -> None:
    """Fail fast on incompatible position-bias / training settings."""
    if INVERSE_PROPENSITY_WEIGHTING and INFERENCE_IMPUTE_LEAKAGE_COLS:
        raise ValueError(
            "INVERSE_PROPENSITY_WEIGHTING and INFERENCE_IMPUTE_LEAKAGE_COLS cannot both be enabled "
            "(IPW drops position from the model; inference uses no null/impute for leakage cols)."
        )


validate_pipeline_config()

# Slim Delta table for eval / SQL analysis (full feature row lives in full_df holdout).
INFERENCE_OUTPUT_COLUMNS = [
    "fanduel_user_id",
    "session_id",
    "game_id",
    "event_date",
    "search_keyword",
    *SCORE_OR_RANK_LEAKAGE_COLS,
    "final_similarity_score",
    "label",
]

# Stable Delta types (avoids DELTA_FAILED_TO_MERGE_FIELDS after schema drift, e.g. double vs int game_position).
INFERENCE_DELTA_CASTS = {
    "game_position": "int",
    "label": "int",
    "click_proba": "double",
    "final_similarity_score": "double",
}
