# casino-search-reranking

Spark XGBoost reranker for casino search: build **`full_df`** from Databricks feature SQL, train on **`label`**, score the holdout week with **`game_position`** nullified.

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
| **Training** | Last **`holdout_period`** days held out in Spark; target **`label`** |
| **Inference** | Holdout slice; **`game_position`** set to null before scoring |

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

Model artifact: **`artifacts/xgb_spark_pipeline/`** (Spark ML Pipeline: encoders + XGBoost). Scores: **`cas_search_reranking_lgbm_inferences`** Delta table with **`click_proba`**.

Optional: tune **`train_sample_fraction`** / **`train_max_rows`** in `constants.py` to shrink the Spark training set.

**Cluster:** your **8 × rgd-fleet.xlarge** setup is appropriate. Use default or **`8g`** for `spark.driver.maxResultSize`; large collects are not needed for train/inference.
