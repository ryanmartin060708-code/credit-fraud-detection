"""
Home.py
-------
Credit Card Fraud Detection Dashboard — Landing Page.
"""

import sys
from pathlib import Path

# Ensure project root is on the Python path when run via `streamlit run app/Home.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.utils import (
    COLOURS,
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    fraud_stats,
    load_artifact,
    load_raw_data,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #6B7280;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .kpi-card {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            color: white;
            box-shadow: 0 4px 15px rgba(99,102,241,0.25);
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
        }
        .kpi-label {
            font-size: 0.85rem;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .status-badge {
            display: inline-block;
            padding: 0.2rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-green { background: #D1FAE5; color: #065F46; }
        .badge-red   { background: #FEE2E2; color: #991B1B; }
        .badge-blue  { background: #DBEAFE; color: #1E40AF; }
        .section-divider { margin: 2rem 0; border: none; border-top: 1px solid #E5E7EB; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ FraudSense AI")
    st.markdown("*An end-to-end ML platform for credit card fraud detection.*")
    st.divider()
    st.markdown("### Navigation")
    st.markdown(
        """
- 🏠 **Home** — You are here
- 📊 **EDA** — Explore the dataset
- 🧠 **Model Training** — Train & tune models
- 📈 **Performance** — Compare model metrics
- 🔍 **SHAP** — Explainability
- 💳 **Predict** — Flag transactions
        """
    )
    st.divider()
    st.caption("Built with Streamlit · Scikit-learn · XGBoost · LightGBM · SHAP")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">🛡️ FraudSense AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Credit Card Fraud Detection · ML Dashboard · Production-Ready</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dataset check
# ---------------------------------------------------------------------------
if not DATA_PATH.exists():
    st.error(
        "⚠️ Dataset not found!  \n"
        f"Place `creditcard.csv` inside `{DATA_PATH.parent}` and refresh.",
        icon="🚨",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load data & cached stats
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset…")
def _load():
    df = load_raw_data()
    return df, fraud_stats(df)


df, stats = _load()

# Check if model is trained
model_ready = MODEL_PATH.exists()
metrics_ready = METRICS_PATH.exists()

if metrics_ready:
    all_metrics = load_artifact(METRICS_PATH)
    from src.evaluate import best_model_name
    best_name = best_model_name(all_metrics)
    best_roc = all_metrics[best_name]["roc_auc"]
else:
    best_name = "N/A"
    best_roc = "—"

# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------
st.markdown("### 📊 Dataset Overview")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Total Transactions", f"{stats['total']:,}")
with c2:
    st.metric("Legitimate", f"{stats['legit']:,}")
with c3:
    st.metric("Fraudulent", f"{stats['fraud']:,}", delta=None)
with c4:
    st.metric("Fraud Rate", f"{stats['fraud_pct']}%")
with c5:
    if model_ready:
        st.metric("Best ROC AUC", best_roc if isinstance(best_roc, str) else f"{best_roc:.4f}")
    else:
        st.metric("Best ROC AUC", "—")

st.divider()

# ---------------------------------------------------------------------------
# Status Indicators
# ---------------------------------------------------------------------------
st.markdown("### 🔧 System Status")
col_a, col_b, col_c = st.columns(3)

with col_a:
    badge = "badge-green" if DATA_PATH.exists() else "badge-red"
    status = "✅ Loaded" if DATA_PATH.exists() else "❌ Missing"
    st.markdown(f'<span class="status-badge {badge}">{status}</span> &nbsp; Dataset', unsafe_allow_html=True)

with col_b:
    badge = "badge-green" if model_ready else "badge-red"
    status = f"✅ {best_name}" if model_ready else "❌ Not Trained"
    st.markdown(f'<span class="status-badge {badge}">{status}</span> &nbsp; Best Model', unsafe_allow_html=True)

with col_c:
    badge = "badge-green" if METRICS_PATH.exists() else "badge-red"
    status = "✅ Available" if METRICS_PATH.exists() else "❌ Not evaluated"
    st.markdown(f'<span class="status-badge {badge}">{status}</span> &nbsp; Evaluation Metrics', unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Quick EDA widgets
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### Class Distribution")
    pie = go.Figure(
        go.Pie(
            labels=["Legitimate", "Fraudulent"],
            values=[stats["legit"], stats["fraud"]],
            hole=0.55,
            marker_colors=[COLOURS["legit"], COLOURS["fraud"]],
            textinfo="label+percent",
        )
    )
    pie.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20),
                      showlegend=False)
    st.plotly_chart(pie, use_container_width=True)

with col_right:
    st.markdown("#### Transaction Amount Distribution")
    import plotly.express as px
    sample = df.sample(min(5000, len(df)), random_state=42)
    hist = px.histogram(
        sample,
        x="Amount",
        color="Class",
        barmode="overlay",
        color_discrete_map={0: COLOURS["legit"], 1: COLOURS["fraud"]},
        labels={"Class": "Type", "Amount": "Amount (USD)"},
        nbins=60,
    )
    hist.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(hist, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Model leaderboard (if trained)
# ---------------------------------------------------------------------------
if metrics_ready:
    st.markdown("### 🏆 Model Leaderboard")
    from src.evaluate import metrics_dataframe
    leaderboard = metrics_dataframe(all_metrics).sort_values("ROC AUC", ascending=False)
    st.dataframe(
        leaderboard.style.highlight_max(color="#D1FAE5", axis=0)
                         .highlight_min(color="#FEE2E2", axis=0)
                         .format("{:.4f}"),
        use_container_width=True,
        height=280,
    )
else:
    st.info(
        "💡 No models trained yet. Head to the **🧠 Model Training** page to get started.",
        icon="ℹ️",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "FraudSense AI · Built with Streamlit, Scikit-learn, XGBoost, LightGBM & SHAP · "
    "Dataset: Kaggle Credit Card Fraud Detection"
)
