"""Shared data pipeline constants."""

full_df_start_date = "2026-07-22"
training_lookback_window = 61
contest_history_lookback_days = training_lookback_window + 30

full_df_table = "sandbox.group_data_science.cas_search_reranking_full_df"
train_df_table = "sandbox.group_data_science.cas_search_reranking_train_df"
inferences_table = "sandbox.group_data_science.cas_search_reranking_lgbm_inferences"
