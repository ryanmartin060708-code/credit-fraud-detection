"""
evaluate.py
-----------
Evaluation helpers: metrics computation and Plotly figure builders for the
Credit Card Fraud Detection pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import COLOURS, METRICS_PATH, get_logger, save_artifact

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """
    Compute a full suite of classification metrics for a fitted model.

    Returns
    -------
    dict with scalar metrics + raw arrays for curve plotting
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_test,
    }
    log.info(
        "%s | Acc=%.4f  P=%.4f  R=%.4f  F1=%.4f  ROC=%.4f  PR=%.4f",
        model_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    )
    return metrics


def evaluate_all(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    persist: bool = True,
) -> dict[str, dict]:
    """Evaluate every model and return a nested metrics dict."""
    all_metrics = {
        name: compute_metrics(model, X_test, y_test, name)
        for name, model in models.items()
    }
    if persist:
        save_artifact(all_metrics, METRICS_PATH)
    return all_metrics


def best_model_name(all_metrics: dict[str, dict], metric: str = "roc_auc") -> str:
    """Return the model name with the highest value for the given metric."""
    return max(all_metrics, key=lambda k: all_metrics[k][metric])


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def fig_confusion_matrix(cm: np.ndarray, model_name: str = "") -> go.Figure:
    """Interactive annotated confusion matrix heatmap."""
    labels = ["Legitimate", "Fraud"]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
        height=400,
    )
    return fig


def fig_roc_curve(all_metrics: dict[str, dict]) -> go.Figure:
    """Overlay ROC curves for all models."""
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
    for name, m in all_metrics.items():
        fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob"])
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={m['roc_auc']:.3f})")
        )
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        legend=dict(x=0.6, y=0.05),
    )
    return fig


def fig_pr_curve(all_metrics: dict[str, dict]) -> go.Figure:
    """Overlay Precision-Recall curves for all models."""
    fig = go.Figure()
    for name, m in all_metrics.items():
        prec, rec, _ = precision_recall_curve(m["y_test"], m["y_prob"])
        fig.add_trace(
            go.Scatter(
                x=rec, y=prec, mode="lines", name=f"{name} (PR-AUC={m['pr_auc']:.3f})"
            )
        )
    fig.update_layout(
        title="Precision-Recall Curves — All Models",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=500,
        legend=dict(x=0.02, y=0.05),
    )
    return fig


def fig_metric_comparison(all_metrics: dict[str, dict]) -> go.Figure:
    """Grouped bar chart comparing key metrics across all models."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels = list(all_metrics.keys())
    fig = go.Figure()
    for metric in metrics_to_plot:
        vals = [all_metrics[m][metric] for m in labels]
        fig.add_trace(go.Bar(name=metric.upper().replace("_", " "), x=labels, y=vals))
    fig.update_layout(
        barmode="group",
        title="Model Metrics Comparison",
        yaxis_title="Score",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_feature_importance(model: Any, feature_names: list[str], model_name: str) -> go.Figure | None:
    """Horizontal bar chart of feature importances (tree-based models only)."""
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:]  # top-20
    fig = go.Figure(
        go.Bar(
            x=importances[idx],
            y=[feature_names[i] for i in idx],
            orientation="h",
            marker_color=COLOURS["accent"],
        )
    )
    fig.update_layout(
        title=f"Top-20 Feature Importances — {model_name}",
        xaxis_title="Importance",
        height=500,
    )
    return fig


def metrics_dataframe(all_metrics: dict[str, dict]) -> pd.DataFrame:
    """Produce a clean comparison DataFrame (no raw arrays)."""
    scalar_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    rows = []
    for name, m in all_metrics.items():
        row = {"Model": name}
        row.update({k.upper().replace("_", " "): m[k] for k in scalar_keys})
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df
