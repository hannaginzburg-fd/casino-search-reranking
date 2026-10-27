# Databricks notebook source
import sys
from pathlib import Path

repo_src = Path("/Workspace/Repos/casino-search-reranking/src")
if not repo_src.exists():
    repo_src = Path.cwd().parent / "src" if (Path.cwd().parent / "src").exists() else Path.cwd() / "src"
sys.path.insert(0, str(repo_src))

# COMMAND ----------

from load_data import load_full_df, save_full_df

SQL_PATH = repo_src / "data" / "training_data.sql"
full_df = load_full_df(SQL_PATH)
save_full_df(full_df)
print(f"full_df: {full_df.shape}")
