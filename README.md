# Agentic Claim Audit System
**Cotiviti Intern Assessment — Topic 2: Clinical Decision Making and Pattern Recognition**

## What This Does
An ML pipeline that detects suspicious Medicare provider billing patterns, 
paired with an LLM-powered audit agent that generates structured investigative 
briefs — ready to hand directly to an SIU investigator.

## Architecture
1. XGBoost classifier scores providers by fraud risk (ROC-AUC: 0.95)
2. SHAP explains which billing features drove each score
3. Claude (claude-sonnet-4-6) generates a 6-section investigative brief per flagged provider
4. Streamlit app displays briefs alongside SHAP charts

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
The app loads pre-generated investigative briefs for 10 flagged providers.  
No API key required to run the Streamlit app after Step 4 is complete.
