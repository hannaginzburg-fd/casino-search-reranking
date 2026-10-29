# casino-search-reranking

Spark XGBoost reranker for casino search: build **`full_df`** from Databricks feature SQL, train on **`label`**, score the holdout week with **`game_position`** and **`final_similarity_score`** handled together (`SCORE_OR_RANK_LEAKAGE_COLS` in `config.py`).

## Layout

```
src/data/training_data.sql   # Feature pipeline → temp view base_features
src/data/constants.py        # Pipeline date / lookback constants
src/data/config.py           # Column rules
src/load_data.py             # Spark SQL → full_df Delta (step 1)
src/features.py              # Feature column list for artifacts
src/train.py                 # SparkXGBClassifier (step 2)
src/inference.py             # Holdout scoring (step 3)
notebooks/load_data.ipynb
notebooks/train.ipynb
notebooks/inference.ipynb
```

## Column rules

| Stage | Behavior |
|-------|----------|
| **`full_df`** | All columns from `base_features` where `final_similarity_score IS NOT NULL` |
| **Training** | `event_date` before validation window; optional **`share_df_summaries`** EDA → `artifacts/training_summaries/` |
| **Validation** | **`validation_days`** immediately before holdout — **`cas_search_reranking_validation_df`** |
| **Inference** | Last **`holdout_period`** days only; **`SCORE_OR_RANK_LEAKAGE_COLS`** nulled on scoring copy |

## Pipeline (run in order)

```bash
cd src
python load_data.py
python train.py
python inference.py
```

**Databricks job:** `load_data.ipynb` → `train.ipynb` → `inference.ipynb`.

**Dependent library (train + inference):** attach PyPI **`xgboost==3.1.1`** on the job cluster. You do not need `%pip install` in the notebooks if the library is on the cluster. Use `%pip` only for interactive runs without job libraries.

Training uses **`SparkXGBClassifier`** on workers (`barrier_mode=True`, `num_workers` from data size). Data stays in Spark — no full-table **`toPandas()`** on train.

Model artifact: **`artifacts/xgb_spark_pipeline/`** (Spark ML Pipeline: encoders + XGBoost). Scores: **`cas_search_reranking_lgbm_inferences`** Delta table (keys, **`game_position`**, **`final_similarity_score`**, **`click_proba`**). Eval SQL: **`src/data/inference_analysis.sql`**.

Optional: tune **`train_sample_fraction`** / **`train_max_rows`** in `constants.py` to shrink the Spark training set.

**Cluster:** **8 × rgd-fleet.xlarge** works for load + Spark XGBoost. Use default or **`8g`** for `spark.driver.maxResultSize`.

**Shared access mode (default):** **`spark_xgb_training = False`** — trains **`xgboost.XGBClassifier`** on a **sampled** subset (`train_max_rows`, default 1M) on the driver, then scores holdout in Spark via **`mapInPandas`** (model copied to **`driver_model_dbfs_path`** under FileStore).

**Single-user / assigned cluster (full data):** **`spark_xgb_training = True`**. **`xgb_num_workers`** defaults to auto-cap (≤ executor cores × instances). **`xgb_barrier_mode = True`** requires that many healthy executors for the whole job.

**`Could not recover from a failed barrier ResultStage`:** usually **`num_workers`** > live executor slots, an executor OOM’d during XGB, or **shared** access mode. Try: set **`xgb_num_workers`** explicitly to executor count (e.g. **`4`**), **`xgb_barrier_mode = False`**, or **`spark_xgb_training = False`** (driver sample). Expand the **Spark UI → failed stage → stderr** for the root error. Turn **`share_df_summaries`** off while debugging train.
