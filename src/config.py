"""Training and inference column rules."""

LABEL_COLUMN = "label"

# Kept in full_df for analysis / joins; dropped before LightGBM fit.
EXCLUDE_FROM_TRAINING = [
    "fanduel_user_id",
    "session_id",
    "game_id",
    "event_date",
    "search_keyword",
    "base_similarity_score",
]

# Not available at serving time — set to null (or omitted) at inference.
INFERENCE_NULL_COLUMNS = ["game_position"]

DEFAULT_MODEL_PATH = "artifacts/search_click_lgbm.txt"
DEFAULT_FEATURE_LIST_PATH = "artifacts/feature_columns.json"
DEFAULT_FULL_DF_PATH = "artifacts/full_df.parquet"
