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
notebooks/load_data.ipynb   # Databricks task 1
notebooks/train.ipynb       # Databricks task 2
notebooks/inference.ipynb   # Databricks task 3
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

**Databricks job:** three tasks in sequence — same commands or `notebooks/load_data.ipynb` → `notebooks/train.ipynb` → `notebooks/inference.ipynb` on a cluster with access to the SQL sources and `sandbox.group_data_science`.

**Recommended: SynapseML on Spark (`use_synapseml = True` in `constants.py`)** — trains LightGBM on **workers**, no multi‑GB `toPandas()` on the driver. Your **8 × rgd-fleet.xlarge (32 GB)** cluster is appropriate for load + train + inference.

**Cluster library (required for SynapseML):** Maven coordinate  
`com.microsoft.azure:synapseml_2.13:1.1.3-spark4.1`  
Repository: `https://mmlspark.blob.core.windows.net/maven`  
Attach to the job cluster and restart. Also add PyPI **`synapseml==1.1.3`** as a dependent library if imports fail.

**Spark config with SynapseML:** you do **not** need a large `spark.driver.maxResultSize` for train/inference. Defaults (or `4g`–`8g`) are fine. **Do not** set `maxResultSize` to `16g` on a **32 GB** driver unless you also raise **`spark.driver.memory`** and accept OOM risk — `maxResultSize` is not a substitute for driver RAM.

**Example job cluster (SynapseML path):**

| Setting | Suggestion |
|--------|------------|
| Driver / workers | `rgd-fleet.xlarge`, 8 workers (as now) |
| `spark.driver.maxResultSize` | **`8g`** or remove override (use default) |
| `spark.driver.memory` | leave default unless load step needs more driver heap |

**Pandas LightGBM fallback (`use_synapseml = False`):** needs **`train_sample_fraction`** / **`train_max_rows`** or a **64 GB+ driver** — not recommended at full data scale on `rgd-fleet.xlarge`.
