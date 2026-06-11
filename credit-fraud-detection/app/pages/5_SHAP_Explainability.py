"""
5_SHAP_Explainability.py
------------------------
SHAP-powered model explainability: summary, waterfall, force plots.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src.explain import (
    compute_shap_values,
    fig_bar,
    fig_force,
    fig_summary,
    fig_waterfall,
    get_explainer,
    top_features_by_shap,
)
from src.preprocess import build_pipeline
from src.train import load_best_model
from src.utils import DATA_PATH, FEATURE_COLS, MODEL_PATH, SCALER_PATH, load_artifact

st.set_page_config(page_title="SHAP · FraudSense", page_icon="🔍", layout="wide")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
if not MODEL_PATH.exists():
    st.warning("No trained model found. Train models first.")
    st.stop()

if not DATA_PATH.exists():
    st.error("Dataset not found.")
    st.stop()

# ---------------------------------------------------------------------------
# Load model & data
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading best model…")
def _model():
    return load_best_model()

@st.cache_data(show_spinner="Loading & preprocessing data for SHAP…")
def _get_test_data():
    pd_data = build_pipeline(persist_scaler=False)
    return pd_data["X_test"], pd_data["y_test"]

model, model_name = _model()
X_test, y_test = _get_test_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔍 SHAP Controls")
    max_shap_rows = st.slider("Samples for SHAP", 100, min(1000, len(X_test)), 300, 50)
    shap_idx = st.slider(
        "Transaction index (for per-sample plots)", 0, max_shap_rows - 1, 0
    )
    st.divider()
    st.caption(f"Active model: **{model_name}**")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔍 SHAP Explainability")
st.markdown(
    f"Understanding **why** the model makes predictions. "
    f"Active model: `{model_name}`"
)
st.divider()

# ---------------------------------------------------------------------------
# Compute SHAP values (cached in session state)
# ---------------------------------------------------------------------------
if "shap_values" not in st.session_state or st.session_state.get("shap_n") != max_shap_rows:
    with st.spinner("Computing SHAP values… (this may take 30–60 seconds for large samples)"):
        explainer = get_explainer(model, X_test)
        shap_values = compute_shap_values(explainer, X_test, max_rows=max_shap_rows)
        st.session_state.shap_values = shap_values
        st.session_state.shap_explainer = explainer
        st.session_state.shap_n = max_shap_rows

shap_values = st.session_state.shap_values
explainer = st.session_state.shap_explainer
X_shap = X_test.iloc[:max_shap_rows]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Summary Plot",
    "📉 Global Bar Chart",
    "💧 Waterfall Plot",
    "⚡ Force Plot",
])

with tab1:
    st.subheader("SHAP Beeswarm Summary Plot")
    st.caption("Each dot = one sample. Colour = feature value (high/low). X-axis = SHAP impact on prediction.")
    fig_s = fig_summary(shap_values, X_shap, max_display=20)
    st.pyplot(fig_s, use_container_width=True)

    st.divider()
    st.markdown("#### Top Features by Mean |SHAP|")
    top_df = top_features_by_shap(shap_values, FEATURE_COLS, top_n=15)
    bar_fig = px.bar(
        top_df, x="Mean |SHAP|", y="Feature", orientation="h",
        color="Mean |SHAP|", color_continuous_scale="Purples",
    )
    bar_fig.update_layout(height=450, yaxis=dict(autorange="reversed"))
    st.plotly_chart(bar_fig, use_container_width=True)

with tab2:
    st.subheader("Global Feature Importance (Mean |SHAP|)")
    st.caption("Average absolute SHAP value per feature across all sampled transactions.")
    fig_b = fig_bar(shap_values, X_shap, max_display=20)
    st.pyplot(fig_b, use_container_width=True)

with tab3:
    st.subheader(f"Waterfall Plot — Transaction #{shap_idx}")
    actual_label = int(y_test.iloc[shap_idx])
    st.markdown(
        f"Actual class: `{'Fraud' if actual_label == 1 else 'Legitimate'}`"
    )
    fig_wf = fig_waterfall(shap_values, idx=shap_idx)
    st.pyplot(fig_wf, use_container_width=True)
    st.caption(
        "Each bar shows how much that feature pushed the prediction "
        "above or below the base rate."
    )

with tab4:
    st.subheader(f"Force Plot — Transaction #{shap_idx}")
    st.caption("Red features push towards fraud; blue features push against it.")
    try:
        html_str = fig_force(explainer, shap_values, X_shap, idx=shap_idx)
        components.html(html_str, height=200, scrolling=True)
    except Exception as e:
        st.warning(f"Force plot could not be rendered: {e}. Try a different index.")

st.divider()
st.caption(
    "SHAP (SHapley Additive exPlanations) provides game-theoretic guarantees of feature attribution. "
    "TreeExplainer is used for tree-based models; KernelExplainer for others."
)
