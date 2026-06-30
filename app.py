import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

DATA_DIR = "./data"

st.set_page_config(
    page_title="Provider Fraud Intelligence System",
    layout="wide",
)


@st.cache_data
def load_artifacts():
    with open(f"{DATA_DIR}/investigative_briefs.json") as f:
        briefs = json.load(f)
    with open(f"{DATA_DIR}/model_artifacts.json") as f:
        meta = json.load(f)
    preds = pd.read_csv(f"{DATA_DIR}/test_predictions.csv")
    shap_df = pd.read_csv(f"{DATA_DIR}/shap_values.csv")
    return briefs, meta, preds, shap_df


briefs, meta, preds, shap_df = load_artifacts()

feature_cols = meta["feature_cols"]
threshold = meta["best_threshold"]

# only show the providers we actually generated briefs for
demo_ids = list(briefs.keys())
demo_preds = (
    preds[preds["Provider"].isin(demo_ids)]
    .sort_values("FraudProb", ascending=False)
    .reset_index(drop=True)
)


def status_label(row):
    if row["FraudFlag"] == 1 and row["Fraud"] == 1:
        return " Confirmed Fraud"
    elif row["FraudFlag"] == 1 and row["Fraud"] == 0:
        return " Flagged for Investigation"
    return " High Confidence Fraud"


def status_color(row):
    if row["FraudFlag"] == 1 and row["Fraud"] == 1:
        return "#d62728"
    elif row["FraudFlag"] == 1 and row["Fraud"] == 0:
        return "#ff7f0e"
    return "#9467bd"


st.markdown("""
<div style='padding: 1rem 0 0.5rem 0'>
    <h1 style='margin-bottom: 0.1rem'> Provider Fraud Intelligence System</h1>
    <p style='color: #888; font-size: 1.05rem; margin-top: 0'>
        AI-powered investigative briefs for provider-level fraud, waste & abuse detection
        &nbsp;|&nbsp; Powered by XGBoost + Claude
    </p>
</div>
<hr style='margin: 0.5rem 0 1.5rem 0'>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC", f"{meta['roc_auc']:.4f}")
c2.metric("PR-AUC", f"{meta['pr_auc']:.4f}")
c3.metric("Fraud F1", f"{meta['best_f1']:.3f}")
c4.metric("Flag Threshold", f"{round(threshold * 100, 1)}%")

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1, 2])

with left:
    st.markdown("### Flagged Providers")
    st.caption(f"{len(demo_preds)} providers · sorted by fraud probability")

    selected = None
    for _, row in demo_preds.iterrows():
        pid = row["Provider"]
        prob = round(row["FraudProb"] * 100, 1)

        if st.button(f"{pid}  —  {prob}%", key=pid, use_container_width=True):
            selected = pid

        st.markdown(
            f"<div style='font-size:0.75rem; color:{status_color(row)}; "
            f"margin:-0.6rem 0 0.4rem 0.3rem'>{status_label(row)}</div>",
            unsafe_allow_html=True
        )

    if selected is None:
        selected = demo_preds.iloc[0]["Provider"]

with right:
    if selected not in briefs:
        st.warning("No investigative brief available for this provider.")
    else:
        entry = briefs[selected]
        row = demo_preds[demo_preds["Provider"] == selected].iloc[0]
        color = status_color(row)

        st.markdown(f"""
        <div style='background:#1a1a2e; border-left: 4px solid {color};
                    padding: 0.8rem 1rem; border-radius: 4px; margin-bottom: 1rem'>
            <span style='font-size:1.2rem; font-weight:700'>
                Provider {selected}
            </span>
            &nbsp;&nbsp;
            <span style='color:{color}; font-weight:600'>{status_label(row)}</span>
            &nbsp;&nbsp;
            <span style='color:#aaa'>Fraud Probability: <b style='color:white'>{entry["fraud_prob"]}%</b></span>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs([" Investigative Brief", " Billing Pattern Analysis"])

        with tab1:
            st.markdown(entry["brief"])

        with tab2:
            match = shap_df[shap_df["Provider"] == selected]
            if match.empty:
                st.info("No billing pattern data available for this provider.")
            else:
                shap_row = match.iloc[0]
                pairs = sorted(
                    ((c, shap_row[c]) for c in feature_cols if c in shap_row),
                    key=lambda p: abs(p[1]),
                    reverse=True
                )[:10]
                names = [p[0].replace("_", " ") for p in pairs][::-1]
                vals = [p[1] for p in pairs][::-1]
                colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]

                fig, ax = plt.subplots(figsize=(7, 4))
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#0e1117")
                ax.barh(names, vals, color=colors, edgecolor="none", height=0.6)
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
    "not findings of fraud on individual claims. Model trained on CMS Medicare data (Kaggle)."
    "</div>",
    unsafe_allow_html=True
)