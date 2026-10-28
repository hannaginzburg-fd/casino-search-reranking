"""Shared data pipeline constants."""

full_df_start_date = "2026-07-22"
training_lookback_window = 61
holdout_period = 7

train_sample_fraction = 1.0  # used when spark_xgb_training=False (driver fit)
train_max_rows = None  # cap rows before driver collect on shared clusters

contest_history_lookback_days = training_lookback_window + 30

xgboost_version = "3.1.1"
# SparkXGBClassifier needs single-user / assigned compute (not shared UC). If False, trains on driver sample.
spark_xgb_training = True
# Executors read this copy during mapInPandas inference (FileStore is cluster-wide).
driver_model_dbfs_path = "/dbfs/FileStore/cas_search_reranking/xgb_model.json"
xgb_num_workers = 8
xgb_barrier_mode = True

schema = "sandbox.group_data_science"

full_df_table = "cas_search_reranking_full_df"
train_df_table = "cas_search_reranking_train_df"
inferences_table = "cas_search_reranking_lgbm_inferences"
