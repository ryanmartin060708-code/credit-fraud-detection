"""
2_Model_Training.py
-------------------
Train, compare, and tune models with live progress bars.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import time

import pandas as pd
import plotly.express as px
import streamlit as st

from src.evaluate import evaluate_all, metrics_dataframe
from src.preprocess import build_pipeline
from src.train import MODEL_NAMES, save_best_model, train_all, tune_model
from src.utils import DATA_PATH, METRICS_PATH, MODEL_PATH, IMBALANCE_METHODS, load_artifact

st.set_page_config(page_title="Training · FraudSense", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Training Settings")

    test_size = st.slider("Test split (%)", 10, 40, 20, 5) / 100
    imbalance_method = st.selectbox("Imbalance strategy", IMBALANCE_METHODS)
    use_class_weight = imbalance_method == "Class Weighting"

    st.divider()
    st.markdown("### Hyperparameter Tuning")
    run_tuning = st.checkbox("Enable Optuna tuning", value=False)
    tune_target = st.selectbox("Tune model", ["XGBoost", "LightGBM"])
    n_trials = st.slider("Optuna trials", 10, 100, 30, 10)
    st.divider()
    st.caption("Tuning adds extra time. Start with all models first.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧠 Model Training")
st.markdown("Train six classifiers on the fraud dataset and compare results in real time.")

if not DATA_PATH.exists():
    st.error("Dataset not found. Place `creditcard.csv` in the `data/` folder.")
    st.stop()

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
for key in ("pipeline_data", "models", "all_metrics"):
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Step 1: Preprocessing
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### Step 1 — Data Preprocessing")

if st.button("⚙️ Run Preprocessing", type="secondary", use_container_width=True):
    with st.spinner(f"Preprocessing with **{imbalance_method}**…"):
        pipeline_data = build_pipeline(test_size=test_size, imbalance_method=imbalance_method)
        st.session_state.pipeline_data = pipeline_data

if st.session_state.pipeline_data:
    pd_data = st.session_state.pipeline_data
    stats = pd_data["stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training samples", f"{stats['train_size']:,}")
    c2.metric("Test samples", f"{stats['test_size']:,}")
    c3.metric("Fraud %", f"{stats['fraud_pct']}%")
    c4.metric("Strategy", stats["imbalance_method"])
    st.success("✅ Preprocessing complete!")
else:
    st.info("Press **Run Preprocessing** to begin.", icon="ℹ️")

# ---------------------------------------------------------------------------
# Step 2: Train all models
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### Step 2 — Train All Models")

if st.button(
    "🚀 Train All Models",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.pipeline_data is None,
):
    pd_data = st.session_state.pipeline_data
    progress_bar = st.progress(0, text="Initialising…")
    log_area = st.empty()
    status_rows = []

    def cb(i, total, name):
        pct = int(i / total * 100)
        progress_bar.progress(pct, text=f"Training {name}…")
        if i > 0 and i <= total:
            status_rows.append(f"✅ {MODEL_NAMES[i-1] if i <= len(MODEL_NAMES) else name}")
            log_area.markdown("\n".join(status_rows))

    models = train_all(
        pd_data["X_train"], pd_data["y_train"],
        class_weight=use_class_weight,
        progress_callback=cb,
    )
    progress_bar.progress(100, text="Evaluating…")

    all_metrics = evaluate_all(models, pd_data["X_test"], pd_data["y_test"], persist=True)

    # Save the best model by ROC AUC
    from src.evaluate import best_model_name
    best_name = best_model_name(all_metrics)
    save_best_model(models[best_name], best_name)

    st.session_state.models = models
    st.session_state.all_metrics = all_metrics
    progress_bar.progress(100, text="Done!")
    st.success(f"✅ All models trained! 🏆 Best model: **{best_name}**")

# ---------------------------------------------------------------------------
# Step 3: Optuna Tuning
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### Step 3 — Hyperparameter Tuning (Optuna)")

if st.button(
    f"🔬 Tune {tune_target} with Optuna",
    type="secondary",
    use_container_width=True,
    disabled=st.session_state.pipeline_data is None or not run_tuning,
):
    if not run_tuning:
        st.warning("Enable Optuna tuning in the sidebar first.")
    else:
        pd_data = st.session_state.pipeline_data
        X_train = pd_data["X_train"]
        y_train = pd_data["y_train"]
        X_test = pd_data["X_test"]
        y_test = pd_data["y_test"]

        # Use test as validation during tuning (acceptable for demonstration)
        bar = st.progress(0, text="Starting Optuna study…")
        best_vals = []

        def tune_cb(i, total, best_val):
            bar.progress(int(i / total * 100), text=f"Trial {i}/{total} · Best F1={best_val:.4f}")
            best_vals.append(best_val)

        best_model, best_params = tune_model(
            tune_target, X_train, y_train, X_test, y_test,
            n_trials=n_trials, progress_callback=tune_cb,
        )
        bar.progress(100, text="Tuning complete!")

        st.success(f"✅ Best params for **{tune_target}**: `{best_params}`")

        # Update metrics with tuned model
        if st.session_state.models is not None:
            st.session_state.models[tune_target] = best_model
            all_metrics = evaluate_all(
                st.session_state.models, pd_data["X_test"], pd_data["y_test"], persist=True
            )
            st.session_state.all_metrics = all_metrics

            from src.evaluate import best_model_name
            best_name = best_model_name(all_metrics)
            save_best_model(st.session_state.models[best_name], best_name)
            st.info(f"🏆 Updated best model: **{best_name}**")

        # Plot Optuna trial progress
        trial_fig = px.line(
            x=list(range(1, len(best_vals) + 1)),
            y=best_vals,
            labels={"x": "Trial", "y": "Best F1 Score"},
            title=f"Optuna Trial Progress — {tune_target}",
        )
        st.plotly_chart(trial_fig, use_container_width=True)

elif not run_tuning:
    st.info("Enable **Optuna tuning** in the sidebar to use this step.", icon="🔬")

# ---------------------------------------------------------------------------
# Results table (loaded from disk or session state)
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### 📊 Model Comparison Table")

all_metrics = st.session_state.all_metrics
if all_metrics is None and METRICS_PATH.exists():
    all_metrics = load_artifact(METRICS_PATH)
    st.session_state.all_metrics = all_metrics

if all_metrics:
    df_results = metrics_dataframe(all_metrics).sort_values("ROC AUC", ascending=False)
    st.dataframe(
        df_results.style.highlight_max(color="#D1FAE5", axis=0)
                        .highlight_min(color="#FEE2E2", axis=0)
                        .format("{:.4f}"),
        use_container_width=True,
    )

    col_l, col_r = st.columns(2)
    with col_l:
        fig_bar = px.bar(
            df_results.reset_index(),
            x="Model", y="F1", color="Model",
            title="F1 Score by Model", height=350,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_r:
        fig_roc = px.bar(
            df_results.reset_index(),
            x="Model", y="ROC AUC", color="Model",
            title="ROC AUC by Model", height=350,
        )
        st.plotly_chart(fig_roc, use_container_width=True)
else:
    st.info("Run training first to see results.", icon="ℹ️")
