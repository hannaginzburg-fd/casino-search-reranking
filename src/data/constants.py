"""Shared data pipeline constants."""

# Training full_df: event_date in [dateadd(-training_lookback_window, end), end] — end is inclusive (data_end_date).
# gameplay_contests.contest_date is a US Pacific calendar day. Snowplow collector_tstamp is UTC.
# SQL derives event_date by converting UTC → Pacific so search rows join contest features on the same day.
gameplay_contest_timezone = "America/Los_Angeles"
# Last included Pacific calendar day for event_date / contest_date filters. Backfill: DATE 'YYYY-MM-DD'.
data_end_date = "DATE '2026-10-28'"
training_lookback_window = 61
holdout_period = 7  # last N days of training full_df → holdout slice
validation_days = 7  # N days immediately before holdout → validation slice; train is the remainder

summary_corr_sample_rows = 100_000  # row cap for feature correlation heatmap

contest_history_lookback_days = training_lookback_window + 30

xgboost_version = "3.1.1"  # Databricks cluster PyPI (train + inference)
spark_xgb_training = True  # False = driver XGB on a sampled train slice
train_sample_fraction = 1.0
train_max_rows = None

schema = "sandbox.group_data_science"

full_df_for_training_table = "cas_search_reranking_full_df_for_training"
full_df_for_inference_table = "cas_search_reranking_full_df_for_inference"
# Train / validation / holdout loaders read the training snapshot table.
full_df_table = full_df_for_training_table
train_df_table = "cas_search_reranking_train_df"
validation_df_table = "cas_search_reranking_validation_df"
inferences_table = "cas_search_reranking_xgboost_inferences"
inference_eval_metrics_table = "cas_search_reranking_inference_eval_metrics"

sample_weight_col = "sample_weight"
