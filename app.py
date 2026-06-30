import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

DATA_DIR = "./data"

st.set_page_config(
    page_title="Provider Fraud Intelligence System",
    page_icon="🔍",
    layout="wide",
)

@st.cache_data
def load_data():
    with open(f"{DATA_DIR}/investigative_briefs.json") as f:
        briefs = json.load(f)
    with open(f"{DATA_DIR}/model_artifacts.json") as f:
        artifacts = json.load(f)
    predictions = pd.read_csv(f"{DATA_DIR}/test_predictions.csv")
    shap_df = pd.read_csv(f"{DATA_DIR}/shap_values.csv")
    return briefs, artifacts, predictions, shap_df

briefs, artifacts, predictions, shap_df = load_data()

FEATURE_COLS = artifacts["feature_cols"]
THRESHOLD = artifacts["best_threshold"]

demo_ids = list(briefs.keys())
demo_preds = (
    predictions[predictions["Provider"].isin(demo_ids)]
    .sort_values("FraudProb", ascending=False)
    .reset_index(drop=True)
)

def status_label(row):
    if row["FraudFlag"] == 1 and row["Fraud"] == 1:
        return "✅ Confirmed Fraud"
    elif row["FraudFlag"] == 1 and row["Fraud"] == 0:
        return "⚠️ Flagged for Investigation"
    else:
        return "🔴 High Confidence Fraud"

def status_color(row):
    if row["FraudFlag"] == 1 and row["Fraud"] == 1:
        return "#d62728"
    elif row["FraudFlag"] == 1 and row["Fraud"] == 0:
        return "#ff7f0e"
    else:
        return "#9467bd"

st.markdown("""
<div style='padding: 1rem 0 0.5rem 0'>
    <h1 style='margin-bottom: 0.1rem'>🔍 Provider Fraud Intelligence System</h1>
    <p style='color: #888; font-size: 1.05rem; margin-top: 0'>
        AI-powered investigative briefs for provider-level fraud, waste & abuse detection
        &nbsp;|&nbsp; Powered by XGBoost + Claude
    </p>
</div>
<hr style='margin: 0.5rem 0 1.5rem 0'>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("ROC-AUC", f"{artifacts['roc_auc']:.4f}")
m2.metric("PR-AUC", f"{artifacts['pr_auc']:.4f}")
m3.metric("Fraud F1", f"{artifacts['best_f1']:.3f}")
m4.metric("Flag Threshold", f"{round(THRESHOLD * 100, 1)}%")

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1, 2])

with left:
    st.markdown("### Flagged Providers")
    st.caption(f"{len(demo_preds)} providers · sorted by fraud probability")

    selected_provider = None

    for _, row in demo_preds.iterrows():
        pid = row["Provider"]
        prob = round(row["FraudProb"] * 100, 1)
        label = status_label(row)
        color = status_color(row)

        if st.button(f"{pid}  —  {prob}%", key=pid, use_container_width=True):
            selected_provider = pid

        st.markdown(
            f"<div style='font-size:0.75rem; color:{color}; "
            f"margin:-0.6rem 0 0.4rem 0.3rem'>{label}</div>",
            unsafe_allow_html=True
        )

    if selected_provider is None:
        selected_provider = demo_preds.iloc[0]["Provider"]

with right:
    if selected_provider not in briefs:
        st.warning("No investigative brief available for this provider.")
    else:
        entry = briefs[selected_provider]
        pred_row = demo_preds[demo_preds["Provider"] == selected_provider].iloc[0]

        prob = entry["fraud_prob"]
        label = status_label(pred_row)
        color = status_color(pred_row)

        st.markdown(f"""
        <div style='background:#1a1a2e; border-left: 4px solid {color};
                    padding: 0.8rem 1rem; border-radius: 4px; margin-bottom: 1rem'>
            <span style='font-size:1.2rem; font-weight:700'>
                Provider {selected_provider}
            </span>
            &nbsp;&nbsp;
            <span style='color:{color}; font-weight:600'>{label}</span>
            &nbsp;&nbsp;
            <span style='color:#aaa'>Fraud Probability: <b style='color:white'>{prob}%</b></span>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📋 Investigative Brief", "📊 Billing Pattern Analysis"])

        with tab1:
            st.markdown(entry["brief"])

        with tab2:
            shap_match = shap_df[shap_df["Provider"] == selected_provider]

            if shap_match.empty:
                st.info("No billing pattern data available for this provider.")
            else:
                shap_row = shap_match.iloc[0]
                shap_vals = {
                    col: shap_row[col]
                    for col in FEATURE_COLS
                    if col in shap_row
                }

                top10 = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                feats = [x[0].replace("_", " ") for x in top10]
                vals = [x[1] for x in top10]
                colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]

                fig, ax = plt.subplots(figsize=(7, 4))
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#0e1117")

                ax.barh(feats[::-1], vals[::-1], color=colors[::-1], edgecolor="none", height=0.6)
                ax.axvline(0, color="#555", linewidth=0.8)
                ax.set_xlabel("SHAP Value  (red = pushes toward fraud)", color="#aaa", fontsize=9)
                ax.tick_params(colors="#ccc", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "Red bars = billing behaviors pushing toward a fraud score. "
                    "Blue bars = behaviors reducing the score. "
                    "Longer bar = stronger influence on this provider's overall risk rating."
                )

st.markdown("<hr style='margin-top:2rem'>", unsafe_allow_html=True)
st.markdown(
    "<div style='color:#555; font-size:0.8rem; text-align:center'>"
    "Decision-support tool only. Scores reflect provider-level billing pattern anomalies — "
    "not findings of fraud on individual claims. "
    "Model trained on CMS Medicare data from Kaggle. "
    "</div>",
    unsafe_allow_html=True
)