"""Training and inference column rules."""

from data.constants import *  # noqa: F403

LABEL_COLUMN = "label"

# Kept in full_df for analysis / joins; dropped before LightGBM fit.
EXCLUDE_FROM_TRAINING = [
    "fanduel_user_id",
    "session_id",
    "game_id",
    "event_date",
    "search_keyword",
    "base_similarity_score",
    "inserted_ts",
]

# Not available at serving time — set to null (or omitted) at inference.
INFERENCE_NULL_COLUMNS = ["game_position"]

DEFAULT_MODEL_PATH = "artifacts/search_click_lgbm.txt"
DEFAULT_SPARK_MODEL_PATH = "artifacts/lightgbm_spark_pipeline"
DEFAULT_FEATURE_LIST_PATH = "artifacts/feature_columns.json"
