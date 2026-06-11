"""
3_Predict.py
------------
Predict individual transactions (manual input) or batch-predict an uploaded CSV.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import io

import pandas as pd
import streamlit as st

from src.predict import predict_batch, predict_single, probability_to_gauge, risk_colour
from src.train import load_best_model
from src.utils import FEATURE_COLS, MODEL_PATH, SCALER_PATH

st.set_page_config(page_title="Predict · FraudSense", page_icon="💳", layout="wide")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model_ready = MODEL_PATH.exists() and SCALER_PATH.exists()

if not model_ready:
    st.error(
        "No trained model found. Train models first on the **🧠 Model Training** page.",
        icon="🚨",
    )
    st.stop()

@st.cache_resource(show_spinner="Loading model…")
def _load_model():
    return load_best_model()

@st.cache_resource(show_spinner="Loading scaler…")
def _load_scaler():
    from src.utils import load_artifact, SCALER_PATH
    return load_artifact(SCALER_PATH)


model, model_name = _load_model()
scaler = _load_scaler()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💳 Transaction Fraud Predictor")
st.markdown(f"Active model: **{model_name}**")
st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_manual, tab_batch = st.tabs(["🔢 Manual Entry", "📁 Batch CSV Upload"])

# =======================================================================
# Manual Entry Tab
# =======================================================================
with tab_manual:
    st.subheader("Enter Transaction Features")
    st.caption(
        "V1–V28 are PCA-transformed features from the original dataset. "
        "Enter values or leave at 0 for the mean."
    )

    col_a, col_b, col_c = st.columns(3)
    cols = [col_a, col_b, col_c]

    feature_inputs: dict[str, float] = {}

    # Time & Amount first
    with col_a:
        feature_inputs["Time"] = st.number_input("Time (seconds)", value=0.0, format="%.2f")
    with col_b:
        feature_inputs["Amount"] = st.number_input("Amount (USD)", value=0.0, min_value=0.0, format="%.2f")

    # V1-V28
    v_features = [f"V{i}" for i in range(1, 29)]
    for j, vf in enumerate(v_features):
        with cols[j % 3]:
            feature_inputs[vf] = st.number_input(vf, value=0.0, format="%.6f")

    st.divider()

    if st.button("🔍 Predict Transaction", type="primary", use_container_width=True):
        with st.spinner("Analysing transaction…"):
            result = predict_single(model, feature_inputs, scaler)

        c1, c2, c3 = st.columns(3)

        if result["prediction"] == 1:
            c1.error(result["label"])
        else:
            c1.success(result["label"])

        c2.metric("Fraud Probability", f"{result['probability'] * 100:.2f}%")
        c3.metric("Confidence", f"{result['confidence']:.1f}%")

        col_badge, col_gauge = st.columns([1, 2])
        with col_badge:
            colour = risk_colour(result["risk_level"])
            st.markdown(
                f"""
                <div style='padding:1.5rem;border-radius:12px;background:{colour}22;
                            border:2px solid {colour};text-align:center;margin-top:1rem'>
                  <div style='font-size:2rem'>{"🚨" if result["prediction"]==1 else "✅"}</div>
                  <div style='font-size:1.4rem;font-weight:700;color:{colour}'>
                    {result["risk_level"]} Risk
                  </div>
                  <div style='color:#6B7280;font-size:0.85rem'>
                    {result["label"]}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_gauge:
            gauge_fig = probability_to_gauge(result["probability"])
            st.plotly_chart(gauge_fig, use_container_width=True)

# =======================================================================
# Batch Upload Tab
# =======================================================================
with tab_batch:
    st.subheader("Batch Prediction — Upload CSV")
    st.markdown(
        "Upload a CSV with the same columns as the training data "
        "(`Time`, `V1`–`V28`, `Amount`). A `Class` column is not required."
    )

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
            st.markdown(f"Loaded **{len(df_upload):,}** rows.")
            st.dataframe(df_upload.head(5), use_container_width=True)

            if st.button("⚡ Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner("Predicting…"):
                    df_result = predict_batch(model, df_upload, scaler)

                fraud_count = int((df_result["Prediction"] == 1).sum())
                legit_count = len(df_result) - fraud_count

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Transactions", len(df_result))
                c2.metric("Flagged as Fraud", fraud_count)
                c3.metric("Fraud Rate", f"{fraud_count/len(df_result)*100:.2f}%")

                st.divider()
                st.dataframe(
                    df_result[
                        ["Time", "Amount", "Prediction_Label", "Fraud_Probability", "Risk_Level"]
                    ].style.applymap(
                        lambda v: "background-color:#FEE2E2" if v == "Fraud" else "",
                        subset=["Prediction_Label"],
                    ),
                    use_container_width=True,
                    height=400,
                )

                # Download
                csv_bytes = df_result.to_csv(index=False).encode()
                st.download_button(
                    "📥 Download Predictions CSV",
                    data=csv_bytes,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Upload a CSV file above to get started.", icon="📁")

        st.markdown("#### Need sample data?")
        if st.button("Generate sample CSV template"):
            sample = pd.DataFrame([{col: 0.0 for col in FEATURE_COLS}])
            st.download_button(
                "📥 Download Template",
                data=sample.to_csv(index=False).encode(),
                file_name="transaction_template.csv",
                mime="text/csv",
            )
