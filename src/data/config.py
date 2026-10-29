"""Training and inference column rules."""

from data.constants import *  # noqa: F403

LABEL_COLUMN = "label"

# Before model fit: write EDA charts under artifacts/training_summaries/ (Spark aggregations; full train slice).
share_df_summaries = True

# Logged slot + baseline search score — same policy for train and serve (see below).
SCORE_OR_RANK_LEAKAGE_COLS = ["game_position"]

# When True, neither column is included as a model feature at all (vs. getting nullified in inference)
EXCLUDE_SCORE_OR_RANK_LEAKAGE_FROM_TRAINING = False

# Kept in full_df for analysis / joins; dropped before model fit.
EXCLUDE_FROM_TRAINING = [
    "fanduel_user_id",
    "session_id",
    "game_id",
    "event_date",
    "search_keyword",
    "base_similarity_score",
    "inserted_ts",
    *([] if not EXCLUDE_SCORE_OR_RANK_LEAKAGE_FROM_TRAINING else SCORE_OR_RANK_LEAKAGE_COLS),
]

# Nulled on the scoring copy only (persisted inference rows keep logged values).
INFERENCE_NULL_COLUMNS = list(SCORE_OR_RANK_LEAKAGE_COLS)

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

XGB_PREDICTION_COL = "click_pred"
XGB_PROBABILITY_COL = "click_proba_vector"

# SparkXGBClassifier early stopping (boolean: True = validation rows, False = train)
XGB_VALIDATION_INDICATOR_COL = "_is_validation"
