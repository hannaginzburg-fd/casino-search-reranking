"""Shared data pipeline constants."""

full_df_start_date = "2026-07-22"
training_lookback_window = 61
holdout_period = 7

# LightGBM runs on the driver. If collect OOMs or hits maxResultSize, lower these (in Spark, before toPandas):
train_sample_fraction = 1.0  # e.g. 0.25 to use ~25% of training rows
train_max_rows = None  # e.g. 500_000 to hard-cap rows (applied after sample)

contest_history_lookback_days = training_lookback_window + 30

xgboost_version = "3.1.1"

schema = "sandbox.group_data_science"

full_df_table = "cas_search_reranking_full_df"
train_df_table = "cas_search_reranking_train_df"
inferences_table = "cas_search_reranking_lgbm_inferences"
