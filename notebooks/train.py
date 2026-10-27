# Databricks notebook source
import sys
from pathlib import Path

repo_src = Path("/Workspace/Repos/casino-search-reranking/src")
if not repo_src.exists():
    repo_src = Path.cwd().parent / "src" if (Path.cwd().parent / "src").exists() else Path.cwd() / "src"
sys.path.insert(0, str(repo_src))

# COMMAND ----------

import train as train_module

train_module.main()
