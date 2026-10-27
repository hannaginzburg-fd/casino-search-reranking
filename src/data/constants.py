"""Shared data pipeline constants."""

full_df_start_date = "2026-07-22"
training_lookback_window = 61
holdout_period = 7
contest_history_lookback_days = training_lookback_window + 30

schema = "sandbox.group_data_science"

full_df_table = "cas_search_reranking_full_df"
train_df_table = "cas_search_reranking_train_df"
inferences_table = "cas_search_reranking_lgbm_inferences"
