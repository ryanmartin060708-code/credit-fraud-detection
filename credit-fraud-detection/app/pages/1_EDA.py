"""
1_EDA.py
--------
Exploratory Data Analysis — interactive Plotly visualisations.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from src.utils import COLOURS, DATA_PATH, TARGET_COL, FEATURE_COLS, fraud_stats, load_raw_data

st.set_page_config(page_title="EDA · FraudSense", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if not DATA_PATH.exists():
    st.error("Dataset not found. Place `creditcard.csv` in the `data/` folder.")
    st.stop()


@st.cache_data(show_spinner="Loading dataset…")
def load():
    df = load_raw_data()
    return df


df = load()
stats = fraud_stats(df)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 EDA Controls")
    sample_pct = st.slider("Chart sample size (%)", 10, 100, 50, 5)
    st.divider()
    st.caption("All charts use Plotly for full interactivity.")

n_sample = int(len(df) * sample_pct / 100)
df_sample = df.sample(n_sample, random_state=42)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Exploratory Data Analysis")
st.markdown(f"Dataset: **{stats['total']:,}** transactions · **{stats['fraud_pct']}%** fraud rate")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "🗂️ Overview",
    "📉 Distributions",
    "⏰ Time Analysis",
    "🔥 Correlations",
    "🔵 PCA",
])

# =======================================================================
# Tab 1 – Overview
# =======================================================================
with tabs[0]:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{stats['total']:,}")
    c2.metric("Features", len(df.columns) - 1)
    c3.metric("Fraud Count", f"{stats['fraud']:,}")
    c4.metric("Fraud %", f"{stats['fraud_pct']}%")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Class Distribution")
        fig_pie = go.Figure(
            go.Pie(
                labels=["Legitimate", "Fraudulent"],
                values=[stats["legit"], stats["fraud"]],
                hole=0.5,
                marker_colors=[COLOURS["legit"], COLOURS["fraud"]],
            )
        )
        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown("#### Fraud vs Legitimate — Bar Chart")
        fig_bar = go.Figure(
            [
                go.Bar(name="Legitimate", x=["Legitimate"], y=[stats["legit"]], marker_color=COLOURS["legit"]),
                go.Bar(name="Fraudulent", x=["Fraudulent"], y=[stats["fraud"]], marker_color=COLOURS["fraud"]),
            ]
        )
        fig_bar.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    with st.expander("📋 Raw dataset preview (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)

    with st.expander("📐 Descriptive statistics"):
        st.dataframe(df.describe(), use_container_width=True)

    with st.expander("🔎 Missing values & dtypes"):
        info_df = pd.DataFrame({
            "dtype": df.dtypes,
            "non-null": df.count(),
            "null": df.isnull().sum(),
            "null %": (df.isnull().mean() * 100).round(2),
        })
        st.dataframe(info_df, use_container_width=True)

# =======================================================================
# Tab 2 – Distributions
# =======================================================================
with tabs[1]:
    st.subheader("Transaction Amount & Feature Distributions")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Amount Distribution (linear)")
        fig = px.histogram(
            df_sample, x="Amount", color="Class",
            color_discrete_map={0: COLOURS["legit"], 1: COLOURS["fraud"]},
            labels={"Class": "Class"}, nbins=80, barmode="overlay",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Amount Distribution (log scale)")
        df_log = df_sample.copy()
        df_log["log_Amount"] = np.log1p(df_log["Amount"])
        fig2 = px.histogram(
            df_log, x="log_Amount", color="Class",
            color_discrete_map={0: COLOURS["legit"], 1: COLOURS["fraud"]},
            labels={"Class": "Class"}, nbins=80, barmode="overlay",
        )
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("#### Boxplot — Amount by Class")
        fig3 = px.box(
            df_sample, x="Class", y="Amount",
            color="Class",
            color_discrete_map={0: COLOURS["legit"], 1: COLOURS["fraud"]},
            labels={"Class": "Class (0=Legit, 1=Fraud)"},
        )
        fig3.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        st.markdown("#### Violin Plot — Amount by Class")
        fig4 = px.violin(
            df_sample, x="Class", y="Amount", color="Class",
            box=True, points="outliers",
            color_discrete_map={0: COLOURS["legit"], 1: COLOURS["fraud"]},
        )
        fig4.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.markdown("#### Individual Feature Distribution")
    feat = st.selectbox("Select feature", FEATURE_COLS, index=0)
    fig5 = px.histogram(
        df_sample, x=feat, color="Class",
        color_discrete_map={0: COLOURS["legit"], 1: COLOURS["fraud"]},
        barmode="overlay", nbins=60,
        title=f"Distribution of {feat}",
    )
    fig5.update_layout(height=380)
    st.plotly_chart(fig5, use_container_width=True)

# =======================================================================
# Tab 3 – Time Analysis
# =======================================================================
with tabs[2]:
    st.subheader("Transaction Time Analysis")
    df_time = df.copy()
    df_time["Hour"] = (df_time["Time"] // 3600) % 24

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Transactions over Time")
        hourly = df_time.groupby(["Hour", TARGET_COL]).size().reset_index(name="Count")
        fig_t = px.line(
            hourly, x="Hour", y="Count", color=TARGET_COL,
            color_discrete_map={0: COLOURS["legit"], 1: COLOURS["fraud"]},
            labels={TARGET_COL: "Class"},
            markers=True,
        )
        fig_t.update_layout(height=380)
        st.plotly_chart(fig_t, use_container_width=True)

    with col_r:
        st.markdown("#### Fraud Count by Hour")
        fraud_hourly = (
            df_time[df_time[TARGET_COL] == 1]
            .groupby("Hour")
            .size()
            .reset_index(name="Fraud Count")
        )
        fig_fh = px.bar(
            fraud_hourly, x="Hour", y="Fraud Count",
            color="Fraud Count", color_continuous_scale="Reds",
        )
        fig_fh.update_layout(height=380)
        st.plotly_chart(fig_fh, use_container_width=True)

# =======================================================================
# Tab 4 – Correlations
# =======================================================================
with tabs[3]:
    st.subheader("Feature Correlation Heatmap")

    with st.spinner("Computing correlations…"):
        corr = df_sample[FEATURE_COLS + [TARGET_COL]].corr()

    fig_hm = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu",
            zmin=-1, zmax=1,
        )
    )
    fig_hm.update_layout(
        title="Pearson Correlation Matrix",
        height=700,
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()
    st.markdown("#### Top Correlated Features with Class")
    top_corr = (
        corr[TARGET_COL]
        .drop(TARGET_COL)
        .abs()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    top_corr.columns = ["Feature", "|Correlation with Class|"]
    fig_corr_bar = px.bar(
        top_corr, x="|Correlation with Class|", y="Feature",
        orientation="h", color="|Correlation with Class|",
        color_continuous_scale="Purples",
    )
    fig_corr_bar.update_layout(height=450, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_corr_bar, use_container_width=True)

# =======================================================================
# Tab 5 – PCA
# =======================================================================
with tabs[4]:
    st.subheader("PCA — 2D Feature Space Visualisation")
    pca_sample_n = st.slider("PCA sample size", 500, min(10000, len(df)), 3000, 500)
    df_pca = df.sample(pca_sample_n, random_state=42)

    with st.spinner("Running PCA…"):
        pca = PCA(n_components=2, random_state=42)
        components = pca.fit_transform(df_pca[FEATURE_COLS].fillna(0))

    df_pca = df_pca.copy()
    df_pca["PC1"] = components[:, 0]
    df_pca["PC2"] = components[:, 1]

    fig_pca = px.scatter(
        df_pca, x="PC1", y="PC2", color=df_pca[TARGET_COL].astype(str),
        color_discrete_map={"0": COLOURS["legit"], "1": COLOURS["fraud"]},
        opacity=0.6, labels={TARGET_COL: "Class"},
        title=f"PCA — Variance explained: {sum(pca.explained_variance_ratio_) * 100:.1f}%",
    )
    fig_pca.update_layout(height=550)
    st.plotly_chart(fig_pca, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("PC1 variance", f"{pca.explained_variance_ratio_[0]*100:.2f}%")
    c2.metric("PC2 variance", f"{pca.explained_variance_ratio_[1]*100:.2f}%")
