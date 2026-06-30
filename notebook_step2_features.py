# rolls up the four claim-level kaggle tables into one row per provider.
# this is the part that actually took the most time to get right - dates,
# sparse diagnosis columns, and the inpatient/outpatient join all needed
# separate handling before anything could go into a model.
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "./data"

print("loading files...")
labels = pd.read_csv(f"{DATA_DIR}/Train-1542865627584.csv")
beneficiary = pd.read_csv(f"{DATA_DIR}/Train_Beneficiarydata-1542865627584.csv")
inpatient = pd.read_csv(f"{DATA_DIR}/Train_Inpatientdata-1542865627584.csv")
outpatient = pd.read_csv(f"{DATA_DIR}/Train_Outpatientdata-1542865627584.csv")

labels["Fraud"] = (labels["PotentialFraud"] == "Yes").astype(int)


def count_filled(df, cols):
    return df[cols].notna().sum(axis=1)


for df, cols in [
    (inpatient, ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]),
    (outpatient, ["ClaimStartDt", "ClaimEndDt"]),
    (beneficiary, ["DOB", "DOD"]),
]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

# --- inpatient ---
print("aggregating inpatient claims...")
diag_cols_inp = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
proc_cols_inp = [f"ClmProcedureCode_{i}" for i in range(1, 7)]

inpatient["claim_duration"] = (inpatient["ClaimEndDt"] - inpatient["ClaimStartDt"]).dt.days.clip(lower=0)
inpatient["los"] = (inpatient["DischargeDt"] - inpatient["AdmissionDt"]).dt.days.clip(lower=0, upper=365)
inpatient["n_diag_codes"] = count_filled(inpatient, diag_cols_inp)
inpatient["n_proc_codes"] = count_filled(inpatient, proc_cols_inp)

inp_agg = inpatient.groupby("Provider").agg(
    inp_claim_count=("ClaimID", "count"),
    inp_avg_reimbursed=("InscClaimAmtReimbursed", "mean"),
    inp_avg_deductible=("DeductibleAmtPaid", "mean"),
    inp_avg_claim_duration=("claim_duration", "mean"),
    inp_avg_los=("los", "mean"),
    inp_avg_diag_codes=("n_diag_codes", "mean"),
    inp_avg_proc_codes=("n_proc_codes", "mean"),
    inp_unique_physicians=("AttendingPhysician", "nunique"),
    inp_unique_beneficiaries=("BeneID", "nunique"),
).reset_index()

inp_agg["inp_unique_physicians_per100"] = inp_agg["inp_unique_physicians"] / inp_agg["inp_claim_count"] * 100
inp_agg = inp_agg.drop(columns=["inp_unique_physicians"])

# --- outpatient ---
print("aggregating outpatient claims...")
diag_cols_out = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]

outpatient["claim_duration"] = (outpatient["ClaimEndDt"] - outpatient["ClaimStartDt"]).dt.days.clip(lower=0)
outpatient["n_diag_codes"] = count_filled(outpatient, diag_cols_out)

out_agg = outpatient.groupby("Provider").agg(
    out_claim_count=("ClaimID", "count"),
    out_avg_reimbursed=("InscClaimAmtReimbursed", "mean"),
    out_avg_deductible=("DeductibleAmtPaid", "mean"),
    out_avg_claim_duration=("claim_duration", "mean"),
    out_avg_diag_codes=("n_diag_codes", "mean"),
    out_unique_physicians=("AttendingPhysician", "nunique"),
    out_unique_beneficiaries=("BeneID", "nunique"),
).reset_index()

out_agg["out_unique_physicians_per100"] = out_agg["out_unique_physicians"] / out_agg["out_claim_count"] * 100
out_agg = out_agg.drop(columns=["out_unique_physicians"])

# --- beneficiary (linked through claims, not joined directly) ---
print("aggregating beneficiary features...")
chronic_cols = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
]
# source data uses 1=yes, 2=no for these - flip to a normal 0/1
for c in chronic_cols:
    beneficiary[c] = (beneficiary[c] == 1).astype(int)

beneficiary["is_deceased"] = beneficiary["DOD"].notna().astype(int)
beneficiary["age"] = (pd.Timestamp("2009-12-01") - beneficiary["DOB"]).dt.days / 365.25
beneficiary["n_chronic_conditions"] = beneficiary[chronic_cols].sum(axis=1)

bene_provider = pd.concat([
    inpatient[["Provider", "BeneID"]],
    outpatient[["Provider", "BeneID"]],
]).drop_duplicates()

bene_with_provider = bene_provider.merge(beneficiary, on="BeneID", how="left")

bene_agg = bene_with_provider.groupby("Provider").agg(
    avg_age=("age", "mean"),
    pct_deceased=("is_deceased", "mean"),
    avg_chronic_conditions=("n_chronic_conditions", "mean"),
    avg_ip_reimbursement=("IPAnnualReimbursementAmt", "mean"),
    avg_op_reimbursement=("OPAnnualReimbursementAmt", "mean"),
).reset_index()

# --- merge everything onto the provider label table ---
print("building provider feature matrix...")
provider_df = labels[["Provider", "Fraud"]].copy()
provider_df = provider_df.merge(inp_agg, on="Provider", how="left")
provider_df = provider_df.merge(out_agg, on="Provider", how="left")
provider_df = provider_df.merge(bene_agg, on="Provider", how="left")

provider_df["inpatient_to_outpatient_ratio"] = (
    provider_df["inp_claim_count"].fillna(0) / (provider_df["out_claim_count"].fillna(0) + 1)
)
provider_df["total_claims"] = provider_df["inp_claim_count"].fillna(0) + provider_df["out_claim_count"].fillna(0)

# providers with no inpatient claims show up as NaN after the merge - that's
# a real "no inpatient activity" signal, not missing data, so it's 0 not dropped
provider_df = provider_df.fillna(0)

print("\n" + "=" * 60)
print("FEATURE MATRIX SUMMARY")
print("=" * 60)
feature_cols = [c for c in provider_df.columns if c not in ("Provider", "Fraud")]
print(f"Shape         : {provider_df.shape}")
print(f"Providers     : {len(provider_df):,}")
print(f"Features      : {len(feature_cols)}")
print(f"Fraud=1       : {provider_df['Fraud'].sum():,} ({provider_df['Fraud'].mean():.2%})")
print(f"NaNs left     : {provider_df.isnull().sum().sum()}")

print("\nfraud vs non-fraud on a few key features:")
key_features = [
    "inp_avg_reimbursed", "inp_avg_los", "out_avg_reimbursed",
    "inp_unique_physicians_per100", "avg_chronic_conditions", "total_claims",
]
for f in key_features:
    fraud_mean = provider_df[provider_df["Fraud"] == 1][f].mean()
    nonfraud_mean = provider_df[provider_df["Fraud"] == 0][f].mean()
    print(f"  {f:<42} fraud={fraud_mean:>8.2f}  non-fraud={nonfraud_mean:>8.2f}")

provider_df.to_csv(f"{DATA_DIR}/provider_features.csv", index=False)
print(f"\nsaved -> {DATA_DIR}/provider_features.csv")