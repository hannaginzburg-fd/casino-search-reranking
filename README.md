# casino-search-reranking

LightGBM reranker for casino search: build **`full_df`** from Databricks feature SQL, train on **`label`**, score at inference with **`game_position`** nullified.

## Layout

```
src/data/training_data.sql   # Feature pipeline → temp view base_features
src/data/constants.py        # Pipeline date / lookback constants
src/data/config.py           # Column rules
src/load_data.py        # Spark SQL → full_df (pandas)
src/features.py         # Train vs inference feature matrices
src/train.py            # Fit + save model
src/predict.py          # Score (game_position nullified)
notebooks/train.py      # Databricks notebook entrypoint
```

## Column rules

| Stage | Behavior |
|-------|----------|
| **`full_df`** | All columns from `base_features` where `final_similarity_score IS NOT NULL` |
| **Training** | Drops `fanduel_user_id`, `session_id`, `game_id`, `event_date`, `search_keyword`, `base_similarity_score`; target is **`label`** |
| **Inference** | Same feature list as training; **`game_position`** set to null before predict |

`game_position` is used in training (known on impressions); it is cleared at inference because rank is not known yet.

## Train (local / Databricks Connect)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd src
python train.py
```

Re-train from cached Delta table (skip feature SQL):

```bash
cd src
python train.py --from-table
```

Writes `full_df` and `train_df` to `sandbox.group_data_science.cas_search_reranking_*` (see `src/data/constants.py`). Each write adds an `inserted_ts` column.

Outputs: `artifacts/search_click_lgbm.txt`, `artifacts/feature_columns.json`, Delta tables above.

## Inference

```bash
cd src
python predict.py
```

Scores land in `sandbox.group_data_science.cas_search_reranking_lgbm_inferences` (with `inserted_ts`).

## Databricks

Clone this repo as a **Repo** under your user workspace, or run `notebooks/train.py` on a cluster with access to the tables in `src/data/training_data.sql`.
