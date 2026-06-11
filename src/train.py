"""
train.py
--------
Model training, hyperparameter optimisation (Optuna), and persistence for
the Credit Card Fraud Detection pipeline.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import (
    METRICS_PATH,
    MODEL_NAMES,
    MODEL_PATH,
    get_logger,
    load_artifact,
    save_artifact,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_model(name: str, class_weight: bool = False, **kwargs) -> Any:
    """Instantiate a model by display name."""
    cw = "balanced" if class_weight else None
    registry = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight=cw, random_state=42, **kwargs
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight=cw, random_state=42, **kwargs
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight=cw, n_jobs=-1, random_state=42, **kwargs
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42, **kwargs
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            **kwargs,
        ),
        "LightGBM": LGBMClassifier(
            class_weight=cw, random_state=42, verbose=-1, **kwargs
        ),
    }
    if name not in registry:
        raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}")
    return registry[name]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_single(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: bool = False,
    **kwargs,
) -> Any:
    """Train a single model and return the fitted estimator."""
    log.info("Training: %s", name)
    model = _build_model(name, class_weight=class_weight, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: bool = False,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Train all models in the registry.

    Parameters
    ----------
    progress_callback : callable | None
        Optional callable(i, total, name) for progress reporting (e.g. Streamlit).

    Returns
    -------
    dict mapping model name → fitted estimator
    """
    models: dict[str, Any] = {}
    total = len(MODEL_NAMES)
    for i, name in enumerate(MODEL_NAMES):
        if progress_callback:
            progress_callback(i, total, name)
        models[name] = train_single(name, X_train, y_train, class_weight=class_weight)
    if progress_callback:
        progress_callback(total, total, "Done")
    log.info("All %d models trained.", total)
    return models


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------

def _xgb_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    model = XGBClassifier(
        **params,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_val)
    return f1_score(y_val, preds)


def _lgbm_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    model = LGBMClassifier(**params, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return f1_score(y_val, preds)


def tune_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 30,
    progress_callback=None,
) -> tuple[Any, dict]:
    """
    Run Optuna hyperparameter search for XGBoost or LightGBM.

    Returns
    -------
    (fitted_best_model, best_params)
    """
    if name not in ("XGBoost", "LightGBM"):
        raise ValueError("Optuna tuning is only supported for XGBoost and LightGBM.")

    objective_fn = _xgb_objective if name == "XGBoost" else _lgbm_objective

    study = optuna.create_study(direction="maximize")

    for i in range(n_trials):
        trial = study.ask()
        value = objective_fn(trial, X_train, y_train, X_val, y_val)
        study.tell(trial, value)
        if progress_callback:
            progress_callback(i + 1, n_trials, study.best_value)

    best_params = study.best_params
    log.info("%s best params: %s  (F1=%.4f)", name, best_params, study.best_value)

    # Retrain on full train set with best params
    best_model = _build_model(name, **best_params)
    best_model.fit(X_train, y_train)
    return best_model, best_params


# ---------------------------------------------------------------------------
# Persist best model
# ---------------------------------------------------------------------------

def save_best_model(model: Any, model_name: str) -> None:
    """Persist the best model to disk along with its display name."""
    save_artifact({"model": model, "name": model_name}, MODEL_PATH)
    log.info("Best model saved: %s", model_name)


def load_best_model() -> tuple[Any, str]:
    """Load the persisted best model. Returns (model, name)."""
    payload = load_artifact(MODEL_PATH)
    return payload["model"], payload["name"]
