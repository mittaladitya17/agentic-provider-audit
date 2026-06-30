# Provider Fraud Intelligence System
**Cotiviti Intern Assessment — Topic 2: Clinical Decision Making and Pattern Recognition**

## What This Does

A two-stage system for detecting and explaining provider-level billing fraud in Medicare claims data.

**Stage 1 — Detection:** An XGBoost classifier scores each provider by how anomalous their overall billing patterns are, using 23 features aggregated from inpatient claims, outpatient claims, and beneficiary data. The model outputs a fraud probability per provider (ROC-AUC: 0.95, PR-AUC: 0.74).

**Stage 2 — Explanation:** For each flagged provider, a Claude-powered agent receives the top SHAP-explained billing anomalies with population comparisons, identifies the most likely fraud scheme, and generates a structured investigative brief ready for SIU review.

> **Scope:** This system operates at the provider level. Scores reflect billing pattern anomalies across a provider's full claims history — not findings on any individual claim.

## Architecture

```
Raw Claims Data (4 CSVs)
        ↓
Feature Engineering (provider-level aggregation)
        ↓
XGBoost Classifier → Fraud Probability per Provider
        ↓
SHAP → Top billing behaviors driving each score
        ↓
Claude Agent → Structured Investigative Brief
        ↓
Streamlit App → Investigator-facing UI
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download from Kaggle: [Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)

Place all four Train CSV files in `./data/`

## Run in Order

```bash
python notebook_step1_profiling.py
python notebook_step2_features.py
python notebook_step3_model.py

export ANTHROPIC_API_KEY="your-key-here"
python notebook_step4_agent.py

streamlit run app.py
```

## Demo

The Streamlit app loads pre-generated investigative briefs for 10 flagged providers. No API key required to run the app after Step 4 completes.
Streamlit app link: https://agentic-claim-audit-nmti7brefbc49reqzyebnn.streamlit.app/

## Repository

GitHub: [github.com/mittaladitya17/agentic-claim-audit](https://github.com/mittaladitya17/agentic-claim-audit)
