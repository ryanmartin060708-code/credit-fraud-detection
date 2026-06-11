"""
utils.py
--------
Shared utility functions for the Credit Card Fraud Detection pipeline.
"""

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "creditcard.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METRICS_PATH = MODELS_DIR / "metrics.pkl"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "fraud_detection") -> logging.Logger:
    """Return a module-level logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_artifact(obj: Any, path: Path) -> None:
    """Persist any Python object to disk via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    get_logger().info("Saved artifact → %s", path)


def load_artifact(path: Path) -> Any:
    """Load a joblib-persisted artifact from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    """Load the raw creditcard CSV. Raises if the file is missing."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place creditcard.csv inside the data/ folder."
        )
    df = pd.read_csv(DATA_PATH)
    get_logger().info("Loaded dataset: %d rows × %d cols", *df.shape)
    return df


def fraud_stats(df: pd.DataFrame) -> dict:
    """Return a dict of key dataset statistics."""
    total = len(df)
    fraud_count = int(df["Class"].sum())
    legit_count = total - fraud_count
    fraud_pct = round(fraud_count / total * 100, 4)
    return {
        "total": total,
        "fraud": fraud_count,
        "legit": legit_count,
        "fraud_pct": fraud_pct,
    }


# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
TARGET_COL: str = "Class"

ALL_FEATURES: list[str] = FEATURE_COLS  # alias used in pages

MODEL_NAMES: list[str] = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
]

IMBALANCE_METHODS: list[str] = ["SMOTE", "Random Undersampling", "Class Weighting"]

# ---------------------------------------------------------------------------
# Colour palette (Plotly-compatible)
# ---------------------------------------------------------------------------

COLOURS = {
    "fraud": "#EF4444",      # red
    "legit": "#3B82F6",      # blue
    "accent": "#8B5CF6",     # violet
    "success": "#10B981",    # green
    "warning": "#F59E0B",    # amber
    "neutral": "#6B7280",    # grey
}
