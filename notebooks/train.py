# Databricks notebook source
import sys
from pathlib import Path

repo_src = Path("/Workspace/Repos/casino-search-reranking/src")
if not repo_src.exists():
    repo_src = Path.cwd().parent / "src" if (Path.cwd().parent / "src").exists() else Path.cwd() / "src"
sys.path.insert(0, str(repo_src))

import pandas as pd

from data.config import DEFAULT_FEATURE_LIST_PATH, DEFAULT_MODEL_PATH
from data.constants import holdout_period
from inference import infer_holdout
from load_data import load_full_df, save_full_df
from train import train

REPO_ROOT = repo_src.parent
SQL_PATH = repo_src / "data" / "training_data.sql"

# COMMAND ----------

full_df = load_full_df(SQL_PATH)
save_full_df(full_df)
print(f"full_df: {full_df.shape}")

# COMMAND ----------

event_date = pd.to_datetime(full_df["event_date"])
holdout_start = event_date.max() - pd.Timedelta(days=holdout_period - 1)
holdout_df = full_df[event_date >= holdout_start]
train_df = full_df[event_date < holdout_start]

metrics = train(
    train_df,
    model_path=REPO_ROOT / DEFAULT_MODEL_PATH,
    feature_list_path=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH,
)
print(metrics)

infer_holdout(
    holdout_df,
    model_path=REPO_ROOT / DEFAULT_MODEL_PATH,
    feature_list_path=REPO_ROOT / DEFAULT_FEATURE_LIST_PATH,
)
