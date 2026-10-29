"""Shared data pipeline constants."""

full_df_start_date = "2026-07-22"
training_lookback_window = 61
holdout_period = 7
validation_days = 7  # calendar days immediately before holdout; used for tuning during train

train_sample_fraction = 1.0  # used when spark_xgb_training=False (driver fit)
train_max_rows = None  # cap rows before driver collect on shared clusters
summary_corr_sample_rows = 100_000  # row cap for feature correlation heatmap

contest_history_lookback_days = training_lookback_window + 30

xgboost_version = "3.1.1"
# SparkXGBClassifier needs single-user / assigned compute (not shared UC). If False, trains on driver sample.
spark_xgb_training = True
# Executors read this copy during mapInPandas inference (FileStore is cluster-wide).
driver_model_dbfs_path = "/dbfs/FileStore/cas_search_reranking/xgb_model.json"
# None = auto-cap to cluster task slots at train time; must be <= live executors if barrier_mode=True
xgb_num_workers = None
# True needs multi-executor cluster with num_workers matching slots; if barrier fails, try False or driver path
xgb_barrier_mode = True
xgb_n_estimators = 500
xgb_early_stopping_rounds = 30  # uses temporal validation slice; stops early to shorten barrier job (spot-safe)

schema = "sandbox.group_data_science"

full_df_table = "cas_search_reranking_full_df"
train_df_table = "cas_search_reranking_train_df"
validation_df_table = "cas_search_reranking_validation_df"
inferences_table = "cas_search_reranking_lgbm_inferences"
