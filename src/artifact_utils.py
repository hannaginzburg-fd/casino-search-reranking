"""Local artifacts/ directory lifecycle (repo-root artifacts, not Delta tables)."""

from __future__ import annotations

import shutil
from pathlib import Path

from data.config import (
    DEFAULT_DRIVER_MODEL_PATH,
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_SPARK_MODEL_PATH,
    DEFAULT_TRAINING_LOSS_PLOT_PATH,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
TRAINING_SUMMARIES_DIR = ARTIFACTS_DIR / "training_summaries"

INFERENCE_ONLY_FILES = (
    "model_feature_importance.csv",
    "holdout_calibration.png",
)


def reset_artifacts_dir() -> None:
    """Remove all prior run outputs under artifacts/ (EDA repopulates training_summaries during train)."""
    if ARTIFACTS_DIR.exists():
        shutil.rmtree(ARTIFACTS_DIR)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def reset_inference_artifacts() -> None:
    """Drop inference-only files so a standalone inference run does not leave stale charts/CSV."""
    for name in INFERENCE_ONLY_FILES:
        path = ARTIFACTS_DIR / name
        if path.is_file():
            path.unlink()


def default_artifact_paths() -> list[Path]:
    """Paths written by train + inference (for docs / tests)."""
    return [
        REPO_ROOT / DEFAULT_TRAINING_LOSS_PLOT_PATH,
        REPO_ROOT / DEFAULT_FEATURE_LIST_PATH,
        REPO_ROOT / DEFAULT_SPARK_MODEL_PATH,
        REPO_ROOT / DEFAULT_DRIVER_MODEL_PATH,
        TRAINING_SUMMARIES_DIR,
        *(ARTIFACTS_DIR / name for name in INFERENCE_ONLY_FILES),
    ]
