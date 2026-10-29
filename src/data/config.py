"""Training and inference column rules."""

from data.constants import *  # noqa: F403

LABEL_COLUMN = "label"

# Logged slot + baseline search score — same policy for train and serve (see below).
SCORE_OR_RANK_LEAKAGE_COLS = ["game_position"]

# When True, neither column is a model feature; both stay on the inference Delta for eval.
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
    "label",
]

DEFAULT_SPARK_MODEL_PATH = "artifacts/xgb_spark_pipeline"
DEFAULT_DRIVER_MODEL_PATH = "artifacts/xgb_model.json"
DEFAULT_FEATURE_LIST_PATH = "artifacts/feature_columns.json"

XGB_PREDICTION_COL = "click_pred"
XGB_PROBABILITY_COL = "click_proba_vector"
