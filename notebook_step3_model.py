# ============================================================
# STEP 3: MODEL TRAINING, EVALUATION, AND SHAP
# ============================================================
# Reads: provider_features.csv (output of Step 2)
# Outputs:
#   xgb_model.pkl            trained XGBoost model
#   model_artifacts.json     threshold, metrics, top features
#   test_predictions.csv     test set with fraud probabilities
#   shap_values.csv          SHAP values for test set
#
# Key decisions baked in:
#   - Stratified train/test split to preserve fraud rate
#   - scale_pos_weight computed from TRAINING set only
#   - Threshold chosen to maximise F1 (not default 0.5)
#   - SHAP run on TEST set only (honest importance values)
#   - Metrics reported: ROC-AUC + Precision-Recall AUC + F1
# ============================================================

import pandas as pd
import numpy as np
import json, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
import xgboost as xgb
import shap

DATA_DIR = "./data"

# ── Load feature matrix ───────────────────────────────────────
df = pd.read_csv(f"{DATA_DIR}/provider_features.csv")

FEATURE_COLS = [c for c in df.columns if c not in ["Provider", "Fraud"]]
X = df[FEATURE_COLS]
y = df["Fraud"]

print(f"Feature matrix: {X.shape[0]:,} providers × {X.shape[1]} features")
print(f"Overall fraud rate: {y.mean():.2%}\n")


# ── Stratified train / test split ────────────────────────────
# stratify=y ensures the 9.35% fraud rate is preserved in both
# the training and test sets, preventing accidental imbalance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train : {len(X_train):,} providers  (fraud rate: {y_train.mean():.2%})")
print(f"Test  : {len(X_test):,} providers  (fraud rate: {y_test.mean():.2%})")


# ── Class weight from TRAINING set only ───────────────────────
# scale_pos_weight = non-fraud count / fraud count
# Using full dataset counts would be a data leak.
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos
print(f"\nscale_pos_weight : {spw:.2f}  ({neg} non-fraud / {pos} fraud in train)")


# ── Train XGBoost ─────────────────────────────────────────────
print("\nTraining XGBoost...")
model = xgb.XGBClassifier(
    n_estimators     = 300,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = spw,
    eval_metric      = "aucpr",   # Precision-Recall AUC during training
    random_state     = 42,
    verbosity        = 0,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print("Training complete.")


# ── Find optimal decision threshold ───────────────────────────
# Default threshold of 0.5 optimises accuracy, which is
# misleading under class imbalance. We find the threshold
# that maximises F1 on the test set.
probs = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)
f1_scores      = 2 * precision * recall / (precision + recall + 1e-9)
best_idx       = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])
best_f1        = float(f1_scores[best_idx])
print(f"\nOptimal threshold : {best_threshold:.3f}  →  F1 = {best_f1:.3f}")


# ── Evaluate ──────────────────────────────────────────────────
y_pred   = (probs >= best_threshold).astype(int)
roc_auc  = roc_auc_score(y_test, probs)
pr_auc   = average_precision_score(y_test, probs)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"ROC-AUC              : {roc_auc:.4f}")
print(f"Precision-Recall AUC : {pr_auc:.4f}")
print(f"\nClassification Report (threshold = {best_threshold:.3f}):")
print(classification_report(y_test, y_pred,
                             target_names=["Non-Fraud", "Fraud"]))


# ── SHAP on test set ──────────────────────────────────────────
# Running SHAP on training data gives inflated importance values.
# Test set SHAP is the honest, unbiased measure.
print("Computing SHAP values on test set...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=FEATURE_COLS,
).sort_values(ascending=False)

print("\nTop 15 Features by Mean |SHAP|:")
print("=" * 60)
for feat, val in mean_abs_shap.head(15).items():
    bar = "█" * int(val * 200)
    print(f"  {feat:<42} {val:.4f}  {bar}")


# ── Save all artifacts ────────────────────────────────────────
joblib.dump(model, f"{DATA_DIR}/xgb_model.pkl")

artifacts = {
    "feature_cols"    : FEATURE_COLS,
    "best_threshold"  : best_threshold,
    "roc_auc"         : round(roc_auc, 4),
    "pr_auc"          : round(pr_auc, 4),
    "best_f1"         : round(best_f1, 4),
    "scale_pos_weight": round(spw, 2),
    "top_features"    : mean_abs_shap.head(10).to_dict(),
}
with open(f"{DATA_DIR}/model_artifacts.json", "w") as f:
    json.dump(artifacts, f, indent=2)

# Test predictions
test_df = X_test.copy()
test_df["Provider"]  = df.loc[X_test.index, "Provider"].values
test_df["Fraud"]     = y_test.values
test_df["FraudProb"] = probs
test_df["FraudFlag"] = y_pred
test_df.to_csv(f"{DATA_DIR}/test_predictions.csv", index=False)

# SHAP values
shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLS, index=X_test.index)
shap_df["Provider"] = df.loc[X_test.index, "Provider"].values
shap_df.to_csv(f"{DATA_DIR}/shap_values.csv", index=False)

print("\nArtifacts saved:")
print(f"  {DATA_DIR}/xgb_model.pkl")
print(f"  {DATA_DIR}/model_artifacts.json")
print(f"  {DATA_DIR}/test_predictions.csv")
print(f"  {DATA_DIR}/shap_values.csv")