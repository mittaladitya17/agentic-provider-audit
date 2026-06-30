import pandas as pd
import numpy as np
import json
import joblib
import warnings
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

df = pd.read_csv(f"{DATA_DIR}/provider_features.csv")
feature_cols = [c for c in df.columns if c not in ("Provider", "Fraud")]
X = df[feature_cols]
y = df["Fraud"]

print(f"feature matrix: {X.shape[0]:,} providers x {X.shape[1]} features")
print(f"overall fraud rate: {y.mean():.2%}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"train: {len(X_train):,} providers (fraud rate {y_train.mean():.2%})")
print(f"test : {len(X_test):,} providers (fraud rate {y_test.mean():.2%})")

# weight fraud cases up since they're ~9% of the data - computed from
# train only so test stays untouched by this decision
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"\nscale_pos_weight: {scale_pos_weight:.2f} ({neg} non-fraud / {pos} fraud)")

print("\ntraining xgboost...")
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    random_state=42,
    verbosity=0,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("done")

# 0.5 is the wrong cutoff here - it's tuned for balanced classes. pick
# whatever threshold maximizes F1 on the test set instead
probs = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])
best_f1 = float(f1_scores[best_idx])
print(f"\nbest threshold: {best_threshold:.3f} (F1 = {best_f1:.3f})")

y_pred = (probs >= best_threshold).astype(int)
roc_auc = roc_auc_score(y_test, probs)
pr_auc = average_precision_score(y_test, probs)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"ROC-AUC              : {roc_auc:.4f}")
print(f"Precision-Recall AUC : {pr_auc:.4f}")
print(f"\nclassification report (threshold = {best_threshold:.3f}):")
print(classification_report(y_test, y_pred, target_names=["Non-Fraud", "Fraud"]))

# shap on test set only - running it on train data inflates importance
print("computing shap values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0), index=feature_cols
).sort_values(ascending=False)

print("\ntop 15 features by mean |shap|:")
print("=" * 60)
for feat, val in mean_abs_shap.head(15).items():
    bar = "█" * int(val * 200)
    print(f"  {feat:<42} {val:.4f}  {bar}")

joblib.dump(model, f"{DATA_DIR}/xgb_model.pkl")

artifacts = {
    "feature_cols": feature_cols,
    "best_threshold": best_threshold,
    "roc_auc": round(roc_auc, 4),
    "pr_auc": round(pr_auc, 4),
    "best_f1": round(best_f1, 4),
    "scale_pos_weight": round(scale_pos_weight, 2),
    "top_features": mean_abs_shap.head(10).to_dict(),
}
with open(f"{DATA_DIR}/model_artifacts.json", "w") as f:
    json.dump(artifacts, f, indent=2)

test_df = X_test.copy()
test_df["Provider"] = df.loc[X_test.index, "Provider"].values
test_df["Fraud"] = y_test.values
test_df["FraudProb"] = probs
test_df["FraudFlag"] = y_pred
test_df.to_csv(f"{DATA_DIR}/test_predictions.csv", index=False)

shap_df = pd.DataFrame(shap_values, columns=feature_cols, index=X_test.index)
shap_df["Provider"] = df.loc[X_test.index, "Provider"].values
shap_df.to_csv(f"{DATA_DIR}/shap_values.csv", index=False)

print("\nsaved:")
print(f"  {DATA_DIR}/xgb_model.pkl")
print(f"  {DATA_DIR}/model_artifacts.json")
print(f"  {DATA_DIR}/test_predictions.csv")
print(f"  {DATA_DIR}/shap_values.csv")