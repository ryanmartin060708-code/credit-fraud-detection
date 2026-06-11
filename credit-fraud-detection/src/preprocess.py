"""
preprocess.py
-------------
Data loading, cleaning, scaling, and train/test splitting for the
Credit Card Fraud Detection pipeline.
"""

from __future__ import annotations

import logging
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import (
    FEATURE_COLS,
    TARGET_COL,
    SCALER_PATH,
    get_logger,
    load_raw_data,
    save_artifact,
)

log = get_logger(__name__)

ImbalanceMethod = Literal["SMOTE", "Random Undersampling", "Class Weighting"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Load raw data (or accept a pre-loaded DataFrame), remove duplicates, and
    handle missing values.

    Parameters
    ----------
    df : pd.DataFrame | None
        If None, data is loaded from disk.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    if df is None:
        df = load_raw_data()

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        log.info("Removed %d duplicate rows.", before - after)

    missing = df.isnull().sum().sum()
    if missing:
        log.warning("%d missing values found; dropping rows.", missing)
        df = df.dropna()

    log.info("Clean dataset: %d rows × %d cols", *df.shape)
    return df


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    persist: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on X_train and transform both splits.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices.
    persist : bool
        Whether to save the fitted scaler to disk.

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    if persist:
        save_artifact(scaler, SCALER_PATH)
    return X_train_scaled, X_test_scaled, scaler


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    log.info(
        "Split → train=%d  test=%d  fraud_train=%.3f%%  fraud_test=%.3f%%",
        len(X_train),
        len(X_test),
        y_train.mean() * 100,
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


def apply_imbalance_strategy(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: ImbalanceMethod = "SMOTE",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply one of three class-imbalance handling strategies to the training set.

    Parameters
    ----------
    method : {"SMOTE", "Random Undersampling", "Class Weighting"}
        "Class Weighting" is a no-op here (handled during model training).

    Returns
    -------
    X_resampled, y_resampled
    """
    if method == "SMOTE":
        sampler = SMOTE(random_state=random_state, k_neighbors=5)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        log.info(
            "SMOTE applied → %d fraud / %d legit",
            y_res.sum(),
            (y_res == 0).sum(),
        )
    elif method == "Random Undersampling":
        sampler = RandomUnderSampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        log.info(
            "Under-sampling applied → %d fraud / %d legit",
            y_res.sum(),
            (y_res == 0).sum(),
        )
    else:
        # "Class Weighting" – no resampling; model receives class_weight param
        log.info("No resampling applied (class weighting handled by model).")
        X_res, y_res = X_train.copy(), y_train.copy()

    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


def build_pipeline(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    imbalance_method: ImbalanceMethod = "SMOTE",
    random_state: int = 42,
    persist_scaler: bool = True,
) -> dict:
    """
    End-to-end preprocessing pipeline.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test, scaler, stats
    """
    df = load_and_clean(df)
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, persist=persist_scaler
    )
    X_train_res, y_train_res = apply_imbalance_strategy(
        X_train_scaled, y_train, imbalance_method, random_state
    )

    stats = {
        "rows": len(df),
        "fraud_pct": round(df[TARGET_COL].mean() * 100, 4),
        "imbalance_method": imbalance_method,
        "train_size": len(X_train_res),
        "test_size": len(X_test_scaled),
    }
    log.info("Preprocessing complete. Stats: %s", stats)

    return {
        "X_train": X_train_res,
        "X_test": X_test_scaled,
        "y_train": y_train_res,
        "y_test": y_test,
        "scaler": scaler,
        "stats": stats,
    }
