"""
4_Model_Performance.py
----------------------
Deep-dive model evaluation: ROC, PR, confusion matrix, classification report,
feature importances — all interactive via Plotly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.evaluate import (
    fig_confusion_matrix,
    fig_feature_importance,
    fig_metric_comparison,
    fig_pr_curve,
    fig_roc_curve,
    metrics_dataframe,
)
from src.train import load_best_model
from src.utils import DATA_PATH, FEATURE_COLS, METRICS_PATH, MODEL_PATH, load_artifact

st.set_page_config(page_title="Performance · FraudSense", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Guard: need trained metrics
# ---------------------------------------------------------------------------
if not METRICS_PATH.exists():
    st.warning("No evaluation data found. Please train models on the **🧠 Model Training** page first.")
    st.stop()

all_metrics = load_artifact(METRICS_PATH)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 Performance Controls")
    model_names = list(all_metrics.keys())
    selected_model = st.selectbox("Inspect model", model_names)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 Model Performance")
st.markdown(f"Comparing **{len(all_metrics)}** trained models.")
st.divider()

# ---------------------------------------------------------------------------
# Overview table
# ---------------------------------------------------------------------------
st.markdown("### 🏆 Model Leaderboard")
df_table = metrics_dataframe(all_metrics).sort_values("ROC AUC", ascending=False)
st.dataframe(
    df_table.style.highlight_max(color="#D1FAE5", axis=0)
                  .highlight_min(color="#FEE2E2", axis=0)
                  .format("{:.4f}"),
    use_container_width=True,
    height=280,
)

st.divider()

# ---------------------------------------------------------------------------
# Metric comparison chart
# ---------------------------------------------------------------------------
st.markdown("### 📊 Metrics Comparison")
fig_cmp = fig_metric_comparison(all_metrics)
st.plotly_chart(fig_cmp, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Curve plots
# ---------------------------------------------------------------------------
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("### ROC Curves")
    fig_roc = fig_roc_curve(all_metrics)
    st.plotly_chart(fig_roc, use_container_width=True)

with col_r:
    st.markdown("### Precision-Recall Curves")
    fig_pr = fig_pr_curve(all_metrics)
    st.plotly_chart(fig_pr, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Per-model deep dive
# ---------------------------------------------------------------------------
st.markdown(f"### 🔬 Deep Dive — {selected_model}")
m = all_metrics[selected_model]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy",  f"{m['accuracy']:.4f}")
c2.metric("Precision", f"{m['precision']:.4f}")
c3.metric("Recall",    f"{m['recall']:.4f}")
c4.metric("F1",        f"{m['f1']:.4f}")
c5.metric("ROC AUC",   f"{m['roc_auc']:.4f}")
c6.metric("PR AUC",    f"{m['pr_auc']:.4f}")

col_cm, col_report = st.columns([1, 1])

with col_cm:
    st.markdown("#### Confusion Matrix")
    fig_cm = fig_confusion_matrix(m["confusion_matrix"], selected_model)
    st.plotly_chart(fig_cm, use_container_width=True)

with col_report:
    st.markdown("#### Classification Report")
    cr = m["classification_report"]
    report_rows = []
    for label, vals in cr.items():
        if isinstance(vals, dict):
            report_rows.append({
                "Class": label,
                "Precision": round(vals.get("precision", 0), 4),
                "Recall":    round(vals.get("recall", 0), 4),
                "F1-Score":  round(vals.get("f1-score", 0), 4),
                "Support":   int(vals.get("support", 0)),
            })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, height=220)

st.divider()

# ---------------------------------------------------------------------------
# Feature importance (tree-based only)
# ---------------------------------------------------------------------------
st.markdown("#### Feature Importances")

# We need the model object — load from disk if available
if MODEL_PATH.exists():
    try:
        best_model, best_name = load_best_model()
        if best_name == selected_model:
            fig_fi = fig_feature_importance(best_model, FEATURE_COLS, selected_model)
            if fig_fi:
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importances not available for this model type.")
        else:
            st.info(
                f"Feature importances are only pre-loaded for the best model ({best_name}). "
                "Re-train to inspect another model."
            )
    except Exception as e:
        st.warning(f"Could not load model for feature importance: {e}")
else:
    st.info("Train models first to see feature importances.")
