"""Shared data pipeline constants."""

full_df_start_date = "2026-07-22"
training_lookback_window = 61
holdout_period = 7

train_sample_fraction = 1.0  # e.g. 0.25 to shrink Spark training rows
train_max_rows = None  # e.g. 500_000 to hard-cap rows (applied after sample)

contest_history_lookback_days = training_lookback_window + 30

xgboost_version = "3.1.1"
# Shared UC clusters: keep barrier_mode False (True needs single-user / assigned access).
xgb_num_workers = 8
xgb_barrier_mode = False

schema = "sandbox.group_data_science"

full_df_table = "cas_search_reranking_full_df"
train_df_table = "cas_search_reranking_train_df"
inferences_table = "cas_search_reranking_lgbm_inferences"
