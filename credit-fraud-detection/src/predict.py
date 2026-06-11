"""
predict.py
----------
Inference utilities for single transactions and batch CSV prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import FEATURE_COLS, SCALER_PATH, get_logger, load_artifact

log = get_logger(__name__)

# Risk thresholds
RISK_LOW = 0.30
RISK_HIGH = 0.70


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _load_scaler() -> StandardScaler:
    return load_artifact(SCALER_PATH)


def predict_single(
    model: Any,
    feature_values: dict[str, float],
    scaler: StandardScaler | None = None,
) -> dict:
    """
    Predict fraud probability for a single transaction.

    Parameters
    ----------
    model : fitted estimator
    feature_values : dict mapping feature name → value
    scaler : fitted StandardScaler (loaded from disk if None)

    Returns
    -------
    dict with keys: prediction (0/1), label, probability, confidence, risk_level
    """
    if scaler is None:
        scaler = _load_scaler()

    row = pd.DataFrame([{col: feature_values.get(col, 0.0) for col in FEATURE_COLS}])
    row_scaled = pd.DataFrame(scaler.transform(row), columns=FEATURE_COLS)

    y_pred = int(model.predict(row_scaled)[0])
    y_prob = float(model.predict_proba(row_scaled)[0, 1])

    risk_level = (
        "Low" if y_prob < RISK_LOW
        else "High" if y_prob >= RISK_HIGH
        else "Medium"
    )

    return {
        "prediction": y_pred,
        "label": "🚨 Fraudulent" if y_pred == 1 else "✅ Legitimate",
        "probability": round(y_prob, 4),
        "confidence": round(max(y_prob, 1 - y_prob) * 100, 2),
        "risk_level": risk_level,
    }


def predict_batch(
    model: Any,
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    """
    Predict fraud for every row in a DataFrame.

    The DataFrame must contain all columns in FEATURE_COLS.
    Returns the original DataFrame augmented with Prediction and Probability columns.
    """
    if scaler is None:
        scaler = _load_scaler()

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    X = df[FEATURE_COLS].copy()
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS, index=X.index)

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    result = df.copy()
    result["Prediction"] = preds
    result["Prediction_Label"] = ["Fraud" if p == 1 else "Legitimate" for p in preds]
    result["Fraud_Probability"] = np.round(probs, 4)
    result["Risk_Level"] = [
        "Low" if p < RISK_LOW else "High" if p >= RISK_HIGH else "Medium" for p in probs
    ]

    log.info(
        "Batch prediction complete: %d rows, %d flagged as fraud (%.2f%%)",
        len(result),
        preds.sum(),
        preds.mean() * 100,
    )
    return result


# ---------------------------------------------------------------------------
# Risk-meter colour helper (used by Streamlit page)
# ---------------------------------------------------------------------------

def risk_colour(risk_level: str) -> str:
    return {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}.get(risk_level, "#6B7280")


def probability_to_gauge(probability: float) -> dict:
    """Return Plotly gauge figure data dict."""
    import plotly.graph_objects as go

    colour = (
        "#10B981" if probability < RISK_LOW
        else "#EF4444" if probability >= RISK_HIGH
        else "#F59E0B"
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(probability * 100, 1),
            title={"text": "Fraud Probability (%)"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": colour},
                "steps": [
                    {"range": [0, 30], "color": "#D1FAE5"},
                    {"range": [30, 70], "color": "#FEF3C7"},
                    {"range": [70, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(height=300)
    return fig
