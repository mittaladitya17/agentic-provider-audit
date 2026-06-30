# quick look at the raw kaggle files before doing any joins/feature work
import pandas as pd
import os

DATA_DIR = "./data"


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
            files["inpatient"] = pd.read_csv(path)
        elif "outpatient" in name:
            files["outpatient"] = pd.read_csv(path)
        elif "train" in name:
            files["labels"] = pd.read_csv(path)
    return files


def profile(files):
    sep = "=" * 60

    print(sep)
    print("SHAPES AND COLUMNS")
    print(sep)
    for name, df in files.items():
        print(f"\n[{name.upper()}]  rows={len(df):,}  cols={df.shape[1]}")
        print(f"  Columns: {list(df.columns)}")

    print(f"\n{sep}")
    print("CLASS BALANCE")
    print(sep)
    labels = files["labels"]
    counts = labels["PotentialFraud"].value_counts()
    print(f"\n  Counts     : {counts.to_dict()}")
    print(f"  Fraud rate : {counts.get('Yes', 0) / len(labels):.2%}")

    print(f"\n{sep}")
    print("MISSING VALUES")
    print(sep)
    for name, df in files.items():
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing):
            print(f"\n[{name.upper()}]")
            for col, cnt in missing.items():
                print(f"  {col}: {cnt:,} missing ({cnt / len(df) * 100:.1f}%)")
        else:
            print(f"\n[{name.upper()}]: no missing values")

    # the inpatient/outpatient overlap with labels matters a lot for step 2 -
    # most providers turn out to be outpatient-only
    print(f"\n{sep}")
    print("JOIN KEY VALIDATION")
    print(sep)
    label_ids = set(files["labels"]["Provider"].astype(str))
    inp_ids = set(files["inpatient"]["Provider"].astype(str))
    out_ids = set(files["outpatient"]["Provider"].astype(str))
    print(f"\n  Label providers     : {len(label_ids):,}")
    print(f"  Inpatient providers : {len(inp_ids):,}  (overlap: {len(label_ids & inp_ids):,})")
    print(f"  Outpatient providers: {len(out_ids):,}  (overlap: {len(label_ids & out_ids):,})")

    print(f"\n{sep}")
    print("DATE COLUMN SAMPLES")
    print(sep)
    for name, df in files.items():
        date_cols = [c for c in df.columns if any(x in c.lower() for x in ("date", "dob", "dod"))]
        if date_cols:
            print(f"\n[{name.upper()}] date cols: {date_cols}")
            print(df[date_cols].head(3).to_string())

    print(f"\n{sep}\nDONE\n{sep}")


if __name__ == "__main__":
    files = load_files(DATA_DIR)
    print(f"Loaded: {list(files.keys())}\n")
    profile(files)