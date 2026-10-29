"""Training and inference column rules."""

from data.constants import *  # noqa: F403

LABEL_COLUMN = "label"

# Kept in full_df for analysis / joins; dropped before model fit.
EXCLUDE_FROM_TRAINING = [
    "fanduel_user_id",
    "session_id",
    "game_id",
    "event_date",
    "search_keyword",
    "base_similarity_score",
    "inserted_ts",
    #"game_position"
]

# Model inputs only — nulled on a scoring copy, not on persisted inference rows.
INFERENCE_NULL_COLUMNS = ["game_position"]

# Slim Delta table for eval / SQL analysis (full feature row lives in full_df holdout).
INFERENCE_OUTPUT_COLUMNS = [
    "fanduel_user_id",
    "session_id",
    "game_id",
    "event_date",
    "search_keyword",
    "game_position",
    "final_similarity_score",
    "label",
]

DEFAULT_SPARK_MODEL_PATH = "artifacts/xgb_spark_pipeline"
DEFAULT_DRIVER_MODEL_PATH = "artifacts/xgb_model.json"
DEFAULT_FEATURE_LIST_PATH = "artifacts/feature_columns.json"

XGB_PREDICTION_COL = "click_pred"
XGB_PROBABILITY_COL = "click_proba_vector"
