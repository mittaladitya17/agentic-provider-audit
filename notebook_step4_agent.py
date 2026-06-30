import pandas as pd
import numpy as np
import json, time
import os
import requests

DATA_DIR = "./data"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
MODEL = "claude-sonnet-4-6"
API_URL = "https://api.anthropic.com/v1/messages"

FEATURE_DESCRIPTIONS = {
    "inp_claim_count": (
        "Total inpatient claims submitted by this provider",
        "Very high volume suggests claim farming or phantom billing"
    ),
    "inp_avg_reimbursed": (
        "Average reimbursement per inpatient claim ($)",
        "Inflated amounts suggest upcoding to more expensive procedures"
    ),
    "inp_avg_deductible": (
        "Average patient deductible per inpatient claim ($)",
        "Unusually low deductibles relative to reimbursement may indicate billing manipulation"
    ),
    "inp_avg_claim_duration": (
        "Average days between inpatient claim start and end date",
        "Inflated durations suggest billing for longer stays than actually occurred"
    ),
    "inp_avg_los": (
        "Average inpatient length of stay (admission to discharge, days)",
        "Fraudsters inflate length of stay to increase DRG reimbursement"
    ),
    "inp_avg_diag_codes": (
        "Average number of diagnosis codes per inpatient claim",
        "Unusually high counts suggest code stuffing to justify higher reimbursement"
    ),
    "inp_avg_proc_codes": (
        "Average number of procedure codes per inpatient claim",
        "High counts may indicate unbundling — billing separately for bundled procedures"
    ),
    "inp_unique_beneficiaries": (
        "Number of unique patients seen for inpatient care",
        "Very high counts relative to claim volume may suggest patient brokering"
    ),
    "inp_unique_physicians_per100": (
        "Unique attending physicians per 100 inpatient claims",
        "High physician cycling suggests a fraud mill using many doctors to spread liability"
    ),
    "out_claim_count": (
        "Total outpatient claims submitted by this provider",
        "Disproportionately high volume is a primary fraud signal"
    ),
    "out_avg_reimbursed": (
        "Average reimbursement per outpatient claim ($)",
        "Elevated amounts suggest upcoding of outpatient visits"
    ),
    "out_avg_deductible": (
        "Average patient deductible per outpatient claim ($)",
        "Anomalous deductible patterns may indicate billing irregularities"
    ),
    "out_avg_claim_duration": (
        "Average days between outpatient claim start and end date",
        "Extended outpatient claim durations are atypical and warrant review"
    ),
    "out_avg_diag_codes": (
        "Average number of diagnosis codes per outpatient claim",
        "Code inflation to justify unnecessary services or higher reimbursement"
    ),
    "out_unique_beneficiaries": (
        "Number of unique patients seen for outpatient care",
        "Unusually high patient volumes may indicate phantom billing"
    ),
    "out_unique_physicians_per100": (
        "Unique attending physicians per 100 outpatient claims",
        "Physician cycling across outpatient claims is a fraud mill indicator"
    ),
    "avg_age": (
        "Average age of this provider's patients (years)",
        "Targeting elderly patients is common in phantom billing and unnecessary services schemes"
    ),
    "pct_deceased": (
        "Proportion of this provider's patients who are deceased",
        "Billing under deceased patient IDs is a direct fraud indicator"
    ),
    "avg_chronic_conditions": (
        "Average number of chronic conditions per patient",
        "Providers treating unusually sick populations may be targeting vulnerable patients"
    ),
    "avg_ip_reimbursement": (
        "Average annual inpatient reimbursement per patient ($)",
        "High per-patient inpatient costs suggest unnecessary admissions"
    ),
    "avg_op_reimbursement": (
        "Average annual outpatient reimbursement per patient ($)",
        "High per-patient outpatient costs suggest unnecessary services or upcoding"
    ),
    "inpatient_to_outpatient_ratio": (
        "Ratio of inpatient to outpatient claims",
        "Abnormal ratio may indicate deliberate misclassification of claim type"
    ),
    "total_claims": (
        "Total claims submitted across inpatient and outpatient",
        "Extreme total volume is the strongest single fraud signal in this dataset"
    ),
}

FRAUD_SCHEME_TAXONOMY = """
- UPCODING: Billing for a more expensive service than was actually provided
- PHANTOM BILLING: Billing for services never rendered, often using real patient IDs
- DECEASED PATIENT BILLING: Submitting claims under the IDs of deceased beneficiaries
- UNNECESSARY SERVICES: Ordering or billing for medically unnecessary procedures
- UNBUNDLING: Billing separately for procedures that should be billed as a single bundled code
- PHYSICIAN CYCLING: Using many different physicians across claims to spread liability and avoid detection
- LENGTH OF STAY INFLATION: Extending reported hospital stays beyond actual discharge to increase DRG payments
"""

print("Loading artifacts...")
features_df = pd.read_csv(f"{DATA_DIR}/provider_features.csv")
predictions = pd.read_csv(f"{DATA_DIR}/test_predictions.csv")
shap_df = pd.read_csv(f"{DATA_DIR}/shap_values.csv")

with open(f"{DATA_DIR}/model_artifacts.json") as f:
    artifacts = json.load(f)

FEATURE_COLS = artifacts["feature_cols"]
THRESHOLD = artifacts["best_threshold"]

pop_means = features_df[FEATURE_COLS].mean().to_dict()

tp = predictions[(predictions["FraudFlag"] == 1) & (predictions["Fraud"] == 1)]
fp = predictions[(predictions["FraudFlag"] == 1) & (predictions["Fraud"] == 0)]
hc = predictions[predictions["Fraud"] == 1].nlargest(5, "FraudProb")

tp_sample = tp.sample(min(5, len(tp)), random_state=42)
fp_sample = fp.sample(min(3, len(fp)), random_state=42)
hc_sample = hc[~hc["Provider"].isin(tp_sample["Provider"])].head(2)

demo_providers = pd.concat([tp_sample, fp_sample, hc_sample]).drop_duplicates("Provider")
print(f"Demo set: {len(demo_providers)} providers  (TP={len(tp_sample)}, FP={len(fp_sample)}, HC={len(hc_sample)})")


def build_prompt(row, shap_row):
    provider_id = row["Provider"]
    prob = round(row["FraudProb"] * 100, 1)
    flagged = "FLAGGED AS HIGH RISK" if row["FraudFlag"] == 1 else "NOT FLAGGED"
    thresh_pct = round(THRESHOLD * 100, 1)

    shap_feats = {
        col: shap_row[col]
        for col in FEATURE_COLS
        if col in shap_row and shap_row[col] > 0
    }
    top5 = sorted(shap_feats.items(), key=lambda x: x[1], reverse=True)[:5]

    feature_lines = []
    for feat, shap_val in top5:
        desc, fraud_note = FEATURE_DESCRIPTIONS.get(feat, (feat, ""))
        prov_val = round(row.get(feat, 0), 2)
        pop_mean = round(pop_means.get(feat, 0), 2)
        feature_lines.append(
            f"  • {desc}\n"
            f"    Provider value: {prov_val}  |  Population mean: {pop_mean}\n"
            f"    Fraud relevance: {fraud_note}"
        )
    feature_block = "\n".join(feature_lines)

    prompt = f"""You are a healthcare fraud investigator assistant at a payment integrity company.
Your role is to produce structured investigative briefs for human SIU investigators.
You are a DECISION-SUPPORT tool — your output guides investigation, it does not determine guilt.

Note: this system operates at the provider level. Scores reflect overall billing pattern anomalies 
across a provider's history — not findings on any individual claim.

PROVIDER ANALYSIS
-----------------
Provider ID      : {provider_id}
Fraud Probability: {prob}%  (flag threshold: {thresh_pct}%)
Model Decision   : {flagged}

TOP BILLING PATTERN ANOMALIES DRIVING THIS SCORE:
{feature_block}

KNOWN FRAUD SCHEME TYPES FOR REFERENCE:
{FRAUD_SCHEME_TAXONOMY}

Generate an investigative brief using EXACTLY these 6 sections with these headers:
1. RISK LEVEL: (one of: High / Medium / Low)
2. FRAUD PROBABILITY: one sentence summarizing the score in plain English
3. PRIMARY RED FLAGS: 3-4 bullet points describing the specific billing anomalies in plain language
4. MOST LIKELY SCHEME: name the scheme from the list above and explain in 2-3 sentences why this provider's pattern fits
5. EVIDENCE TO PURSUE: 3-4 specific records or documents an investigator should request
6. RECOMMENDED ACTION: one clear, specific next step

Rules:
- Write for a fraud investigator, not a data scientist
- Reference the provider's actual numbers when making claims
- Keep the entire brief under 400 tokens
- Do not invent information not present in the data above
- Do not render a verdict — this is investigative guidance only"""

    return prompt


def call_claude(prompt):
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": MODEL,
        "max_tokens": 600,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
    except Exception as e:
        return f"[API ERROR: {str(e)}]"


print(f"\nGenerating investigative briefs for {len(demo_providers)} providers...")
print("(One API call per provider — expect ~30 seconds total)\n")

briefs = {}

for _, row in demo_providers.iterrows():
    provider_id = row["Provider"]

    shap_match = shap_df[shap_df["Provider"] == provider_id]
    if shap_match.empty:
        print(f"  [{provider_id}] — no data, skipping")
        continue
    shap_row = shap_match.iloc[0].to_dict()

    feat_match = features_df[features_df["Provider"] == provider_id]
    if not feat_match.empty:
        for col in FEATURE_COLS:
            if col not in row:
                row[col] = feat_match.iloc[0][col]

    prompt = build_prompt(row, shap_row)
    brief = call_claude(prompt)

    briefs[provider_id] = {
        "provider_id": provider_id,
        "fraud_prob": round(float(row["FraudProb"]) * 100, 1),
        "fraud_flag": int(row["FraudFlag"]),
        "actual_fraud": int(row["Fraud"]),
        "brief": brief,
    }

    status = (
        "TRUE POSITIVE" if row["FraudFlag"] == 1 and row["Fraud"] == 1
        else "FALSE POSITIVE" if row["FraudFlag"] == 1 and row["Fraud"] == 0
        else "HIGH CONFIDENCE"
    )
    print(f"  [{provider_id}]  prob={round(row['FraudProb']*100,1)}%  {status}")
    print(f"  {brief[:120].strip()}...")
    print()

    time.sleep(0.5)

with open(f"{DATA_DIR}/investigative_briefs.json", "w") as f:
    json.dump(briefs, f, indent=2)

print(f"\nSaved {len(briefs)} briefs → {DATA_DIR}/investigative_briefs.json")