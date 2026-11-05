# casino-search-reranking

Spark XGBoost reranker for casino search: build **`full_df`** from Databricks feature SQL, train on **`label`**, score the holdout week with **`game_position`** and **`final_similarity_score`** handled together (`SCORE_OR_RANK_LEAKAGE_COLS` in `config.py`).

## Layout

```
src/data/data_for_training.sql    # Historical feature pipeline → base_features (train/val/holdout)
src/data/data_for_inference.sql   # Same features; event_date = current_date() only
src/data/constants.py             # Pipeline date / lookback constants
src/data/config.py                # Column rules
src/load_data.py                  # Shared Delta loaders / train-holdout splits
src/load_data_for_training.py     # SQL → cas_search_reranking_full_df_for_training
src/load_data_for_inference.py    # SQL → cas_search_reranking_full_df_for_inference (daily)
src/features.py                   # Feature column list for artifacts
src/train.py                      # SparkXGBClassifier (step 2)
src/inference.py                  # Holdout scoring (step 3)
notebooks/load_data_for_training.ipynb
notebooks/load_data_for_inference.ipynb
notebooks/train.ipynb
notebooks/inference.ipynb
```

## Delta tables

| Table | Purpose |
|-------|---------|
| **`sandbox.group_data_science.cas_search_reranking_full_df_for_training`** | Multi-day history for train / validation / holdout |
| **`sandbox.group_data_science.cas_search_reranking_full_df_for_inference`** | Today’s search impressions + features for production scoring |

## Column rules

| Stage | Behavior |
|-------|----------|
| **Training full_df** | `event_date` in **`[end − training_lookback_window, end)`** with **`training_data_end_date`** (default `current_date()`, so **today excluded**); rows need `final_similarity_score IS NOT NULL` |
| **Holdout / validation** | From that table: **last 7 days** = holdout; **prior 7 days** = validation; earlier = train (`holdout_period`, `validation_days`) |
| **Training** | `event_date` before validation window; optional **`share_df_summaries`** EDA → `artifacts/training_summaries/<split>/` (`share_df_summaries_splits`: train, validation, holdout) |
| **Validation** | **`validation_days`** immediately before holdout — **`cas_search_reranking_validation_df`** |
| **Training (IPW)** | If **`INVERSE_PROPENSITY_WEIGHTING`**: drop **`game_position`** from features; train with **`sample_weight = 1/P(position)`** (clipped); mutually exclusive with inference null/impute |
| **Inference (eval job)** | Standard score (no IPW weights); with IPW, no null/impute on leakage cols; logged cols unchanged in Delta |
| **Daily inference load** | Snowplow + `base_features` filtered to **`event_date = current_date()`**; rolling R1/R7/etc. still use **`contest_history_lookback_days`** |

## Pipeline (run in order)

**Model training / offline eval:**

```bash
cd src
python load_data_for_training.py
python train.py
python inference.py
```

**Daily feature refresh for serving:**

```bash
cd src
python load_data_for_inference.py
```

**Databricks jobs:** `load_data_for_training.ipynb` → `train.ipynb` → `inference.ipynb` for offline pipeline; schedule `load_data_for_inference.ipynb` for daily features.

**Dependent library (train + inference):** attach PyPI **`xgboost==3.1.1`** on the job cluster. You do not need `%pip install` in the notebooks if the library is on the cluster. Use `%pip` only for interactive runs without job libraries.

Training uses **`SparkXGBClassifier`** on workers (`barrier_mode=True`, `num_workers` from data size). Data stays in Spark — no full-table **`toPandas()`** on train.

Model artifact: **`artifacts/xgb_spark_pipeline/`** (Spark ML Pipeline: encoders + XGBoost). Scores: **`cas_search_reranking_xgboost_inferences`** Delta table (**append** per run; **`inserted_ts`** = write time). Eval SQL: **`src/data/inference_analysis.sql`** (filter latest run with `inserted_ts` if needed).

Optional: tune XGB hyperparameters at the top of **`src/data/config.py`**. Cluster/runtime flags (`spark_xgb_training`, `train_max_rows`, …) live in **`constants.py`**.

**Cluster:** **8 × rgd-fleet.xlarge** works for load + Spark XGBoost. Use default or **`8g`** for `spark.driver.maxResultSize`.

**Shared access mode (default):** **`spark_xgb_training = False`** — trains **`xgboost.XGBClassifier`** on a **sampled** subset (`train_max_rows`, default 1M) on the driver, then scores holdout in Spark via **`mapInPandas`** (model copied to **`driver_model_dbfs_path`** under FileStore).

**Single-user / assigned cluster (full data):** **`spark_xgb_training = True`**. **`xgb_num_workers`** defaults to auto-cap (≤ executor cores × instances). **`xgb_barrier_mode = True`** requires that many healthy executors for the whole job.

**Barrier / `collectAndServe` failures:** open **Spark UI → failed stage** for the real cause.

| Root cause | What to do |
|------------|------------|
| **`spot instance preemption`** (your log: `ExecutorLostFailure … spot instance kill`) | Barrier XGB **cannot survive** losing an executor mid-job. Use a **job cluster with on-demand workers only** (disable spot / “use spot instances”), or **`spark_xgb_training = False`** on the driver. |
| **`num_workers`** too high | Set **`xgb_num_workers`** to executor cores × instances, or leave **`None`** for auto-cap. |
| **Shared** access mode | **`spark_xgb_training = False`**. |

**`xgb_early_stopping_rounds`** (default 30) fits on train + temporal validation and often stops **before** 500 trees—less time exposed to spot kills. Still use on-demand for long Spark XGB runs.

### Spark XGB checklist (train + inference)

| Item | Notes |
|------|--------|
| **Cluster** | Single-user / assigned (not shared UC); **on-demand workers** for train; PyPI **`xgboost==3.1.1`** on both job clusters. |
| **`num_workers`** | **`None`** = auto-cap to executor cores × instances; set explicitly if auto estimate is wrong on dynamic clusters. |
| **Validation indicator** | **`_is_validation`**: **`boolean`** (`False` train / `True` val); inference adds **`False`**. |
| **Label / bools** | **`label`** cast to **double**; booleans cast to **double** before the ML pipeline (matches XGB). |
| **Artifact path** | Train writes **`artifacts/xgb_spark_pipeline/`** on the driver; inference cluster must read the same path (Repo Files / workspace git / DBFS copy). |
| **Imbalance** | **`xgb_auto_scale_pos_weight`** sets **`scale_pos_weight`** from train click rate. |
| **Memory** | ~31M train rows / **`num_workers`** ≈ millions of rows per XGB worker—size executors accordingly or reduce data / workers. |
