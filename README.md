# casino-search-reranking

LightGBM reranker for casino search: build **`full_df`** from Databricks feature SQL, train on **`label`**, score at inference with **`game_position`** nullified.

## Layout

```
src/data/training_data.sql   # Feature pipeline → temp view base_features
src/data/constants.py        # Pipeline date / lookback constants
src/data/config.py           # Column rules
src/load_data.py        # Spark SQL → full_df Delta (step 1)
src/features.py         # Train vs inference feature matrices
src/train.py            # Fit + save model from Delta (step 2)
src/inference.py        # Score holdout week (step 3)
notebooks/load_data.py  # Databricks task 1
notebooks/train.py      # Databricks task 2
notebooks/inference.py  # Databricks task 3
```

## Column rules

| Stage | Behavior |
|-------|----------|
| **`full_df`** | All columns from `base_features` where `final_similarity_score IS NOT NULL` |
| **Training** | Last **`holdout_period`** days (see `constants.py`) held out; drops id/date/keyword columns; target is **`label`** |
| **Inference** | Same feature list as training on the holdout slice; **`game_position`** set to null before scoring |

`game_position` is used in training (known on impressions); it is cleared at inference because rank is not known yet.

## Pipeline (run in order)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd src
python load_data.py    # 1. feature SQL → full_df Delta
python train.py        # 2. train (excludes last holdout_period days) → train_df Delta + artifacts/
python inference.py    # 3. score holdout week → lgbm_inferences Delta
```

Table names and lookbacks live in `src/data/constants.py`. Each Delta write adds `inserted_ts`.

**Databricks job:** three tasks in sequence — same commands or `notebooks/load_data.py` → `notebooks/train.py` → `notebooks/inference.py` on a cluster with access to the SQL sources and `sandbox.group_data_science`.
