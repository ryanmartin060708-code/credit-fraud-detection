"""
explain.py
----------
SHAP-based model explainability utilities.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Explainer factory
# ---------------------------------------------------------------------------

def get_explainer(model: Any, X_background: pd.DataFrame) -> shap.Explainer:
    """
    Return the most appropriate SHAP explainer for the given model type.

    Tree-based models use TreeExplainer; others fall back to KernelExplainer
    on a small background sample.
    """
    tree_types = (
        "XGBClassifier",
        "LGBMClassifier",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "DecisionTreeClassifier",
    )
    model_class = type(model).__name__
    if model_class in tree_types:
        log.info("Using TreeExplainer for %s", model_class)
        return shap.TreeExplainer(model)
    else:
        log.info("Using KernelExplainer for %s (slow—using 100 background rows)", model_class)
        background = shap.sample(X_background, 100)
        return shap.KernelExplainer(model.predict_proba, background)


def compute_shap_values(
    explainer: shap.Explainer,
    X: pd.DataFrame,
    max_rows: int = 500,
) -> shap.Explanation | np.ndarray:
    """
    Compute SHAP values for up to *max_rows* rows of X.

    Returns a SHAP Explanation object when available, otherwise a raw array.
    """
    X_sample = X.iloc[:max_rows]
    log.info("Computing SHAP values for %d samples…", len(X_sample))
    sv = explainer(X_sample)
    return sv


# ---------------------------------------------------------------------------
# Matplotlib figure builders (returned as plt.Figure for Streamlit)
# ---------------------------------------------------------------------------

def fig_summary(shap_values, X: pd.DataFrame, max_display: int = 20) -> plt.Figure:
    """Beeswarm / bar summary plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.sca(ax)
    shap.summary_plot(shap_values, X.iloc[:len(shap_values)], max_display=max_display, show=False)
    plt.tight_layout()
    return fig


def fig_bar(shap_values, X: pd.DataFrame, max_display: int = 20) -> plt.Figure:
    """Global mean |SHAP| bar chart."""
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.sca(ax)
    shap.summary_plot(
        shap_values, X.iloc[:len(shap_values)], plot_type="bar",
        max_display=max_display, show=False
    )
    plt.tight_layout()
    return fig


def fig_waterfall(shap_values, idx: int = 0) -> plt.Figure:
    """Waterfall plot for a single prediction."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.waterfall_plot(shap_values[idx], show=False)
    plt.tight_layout()
    return fig


def fig_force(explainer: shap.Explainer, shap_values, X: pd.DataFrame, idx: int = 0):
    """Force plot HTML (returned as an HTML string for st.components)."""
    html = shap.force_plot(
        explainer.expected_value if hasattr(explainer, "expected_value") else 0,
        shap_values[idx].values if hasattr(shap_values[idx], "values") else shap_values[idx],
        X.iloc[idx],
        matplotlib=False,
    )
    shap_html = f"<head>{shap.getjs()}</head><body>{html.html()}</body>"
    return shap_html


def top_features_by_shap(shap_values, feature_names: list[str], top_n: int = 10) -> pd.DataFrame:
    """
    Return a DataFrame of mean |SHAP| per feature, sorted descending.
    """
    if hasattr(shap_values, "values"):
        vals = shap_values.values
    else:
        vals = shap_values

    # Handle multi-output (take class-1 slice if needed)
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    mean_abs = np.abs(vals).mean(axis=0)
    df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs})
    return df.sort_values("Mean |SHAP|", ascending=False).head(top_n).reset_index(drop=True)
