"""Training and inference column rules."""

from data.constants import *  # noqa: F403

LABEL_COLUMN = "label"

##### 3 Options for Handling Position Bias: #####
# 1) Nullifying game position at inference time
# 2) Mean-imputing game position at inference time
# 3) IPW weighting - inversely weight each labeled row in *training*; inference unchanged

# Position-bias IPW: drop position from features, row-weight train loss; no inference leakage handling.
INVERSE_PROPENSITY_WEIGHTING = True
IPW_POSITION_COLUMN = "game_position"

# Before model fit: write EDA charts under artifacts/training_summaries/ (Spark aggregations; full train slice).
share_df_summaries = True

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
    "base_similarity_score",
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

DEFAULT_SPARK_MODEL_PATH = "artifacts/xgb_spark_pipeline"
DEFAULT_DRIVER_MODEL_PATH = "artifacts/xgb_model.json"
DEFAULT_FEATURE_LIST_PATH = "artifacts/feature_columns.json"
DEFAULT_TRAINING_LOSS_PLOT_PATH = "artifacts/training_logloss.png"

XGB_PREDICTION_COL = "click_pred"
XGB_PROBABILITY_COL = "click_proba_vector"

# SparkXGBClassifier early stopping (boolean: True = validation rows, False = train)
XGB_VALIDATION_INDICATOR_COL = "_is_validation"
