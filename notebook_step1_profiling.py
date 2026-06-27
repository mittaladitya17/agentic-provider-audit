# ============================================================
# STEP 1: DATA PROFILING
# ============================================================
# Run this first. It loads all four CSV files and gives you
# a complete inventory before touching anything.
# ------------------------------------------------------------
# Expected files in your data directory:
#   Train-1542865627584.csv                   (provider labels)
#   Train_Beneficiarydata-1542865627584.csv   (patient info)
#   Train_Inpatientdata-1542865627584.csv     (inpatient claims)
#   Train_Outpatientdata-1542865627584.csv    (outpatient claims)
# ============================================================

import pandas as pd
import numpy as np
import os

# ── CHANGE THIS to wherever your CSV files are ───────────────
DATA_DIR = "./data"
# ─────────────────────────────────────────────────────────────

def load_files(data_dir):
    files = {}
    for f in os.listdir(data_dir):
        if not f.endswith(".csv"):
            continue
        path = os.path.join(data_dir, f)
        name = f.lower()
        if "beneficiary" in name:
            files["beneficiary"] = pd.read_csv(path)
        elif "inpatient" in name:
            files["inpatient"]   = pd.read_csv(path)
        elif "outpatient" in name:
            files["outpatient"]  = pd.read_csv(path)
        elif "train" in name:
            files["labels"]      = pd.read_csv(path)
    return files


def profile(files):
    sep = "=" * 60

    # 1. Shape and column names
    print(sep)
    print("SHAPES AND COLUMNS")
    print(sep)
    for name, df in files.items():
        print(f"\n[{name.upper()}]  rows={len(df):,}  cols={df.shape[1]}")
        print(f"  Columns: {list(df.columns)}")

    # 2. Class balance
    print(f"\n{sep}")
    print("CLASS BALANCE")
    print(sep)
    lb  = files["labels"]
    col = "PotentialFraud"
    counts = lb[col].value_counts()
    print(f"\n  Label column : '{col}'")
    print(f"  Counts       : {counts.to_dict()}")
    print(f"  Fraud rate   : {counts.get('Yes', 0) / len(lb):.2%}")

    # 3. Missing values
    print(f"\n{sep}")
    print("MISSING VALUES")
    print(sep)
    for name, df in files.items():
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing):
            print(f"\n[{name.upper()}]")
            for col, cnt in missing.items():
                print(f"  {col}: {cnt:,} missing ({cnt/len(df)*100:.1f}%)")
        else:
            print(f"\n[{name.upper()}]: no missing values")

    # 4. Join key validation
    print(f"\n{sep}")
    print("JOIN KEY VALIDATION")
    print(sep)
    lb_providers  = set(files["labels"]["Provider"].astype(str))
    inp_providers = set(files["inpatient"]["Provider"].astype(str))
    out_providers = set(files["outpatient"]["Provider"].astype(str))
    print(f"\n  Label providers     : {len(lb_providers):,}")
    print(f"  Inpatient providers : {len(inp_providers):,}  "
          f"(overlap with labels: {len(lb_providers & inp_providers):,})")
    print(f"  Outpatient providers: {len(out_providers):,}  "
          f"(overlap with labels: {len(lb_providers & out_providers):,})")

    # 5. Date column samples
    print(f"\n{sep}")
    print("DATE COLUMN SAMPLES")
    print(sep)
    for name, df in files.items():
        date_cols = [c for c in df.columns
                     if any(x in c.lower() for x in ["date","dob","dod"])]
        if date_cols:
            print(f"\n[{name.upper()}] date cols: {date_cols}")
            print(df[date_cols].head(3).to_string())

    print(f"\n{sep}")
    print("PROFILING COMPLETE")
    print(sep)


# ── Run ───────────────────────────────────────────────────────
files = load_files(DATA_DIR)
print(f"Files loaded: {list(files.keys())}\n")
profile(files)