"""Read-only initial audit for the CarWise CSV."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "car_details_v3.csv"
TARGET_COLUMN = "selling_price"

def audit_dataset():
    print("\\n========== CARWISE DATASET AUDIT ==========")
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found: {DATA_PATH}")
        print("Place the CSV in data/raw/ as car_details_v3.csv.")
        return

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        print(f"Could not read CSV: {exc}")
        return

    print("\\n1. SHAPE")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")

    print("\\n2. COLUMNS")
    print(df.columns.tolist())

    print("\\n3. FIRST FIVE ROWS")
    print(df.head().to_string())

    print("\\n4. DATA TYPES AND NON-NULL COUNTS")
    df.info()

    print("\\n5. MISSING VALUES")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (df.isna().mean() * 100).round(2)
    })
    print(missing.to_string())

    print("\\n6. EXACT DUPLICATE ROWS")
    print(f"Count: {df.duplicated().sum():,}")

    print("\\n7. NUMERICAL SUMMARY")
    numeric = df.describe(include="number").T
    print(numeric.to_string() if not numeric.empty else "No numeric columns.")

    print("\\n8. CATEGORICAL SUMMARY")
    categorical = df.describe(include=["object", "category"]).T
    print(categorical.to_string() if not categorical.empty else "No categorical columns.")

    print("\\n9. TOP CATEGORY VALUES")
    for column in df.select_dtypes(include=["object", "category"]).columns:
        print(f"\\n{column} — unique: {df[column].nunique(dropna=True)}")
        print(df[column].value_counts(dropna=False).head(10).to_string())

    print("\\n10. TARGET CHECK")
    if TARGET_COLUMN in df.columns:
        print(df[TARGET_COLUMN].describe().to_string())
        print("Missing target values:", df[TARGET_COLUMN].isna().sum())
    else:
        print(f"WARNING: '{TARGET_COLUMN}' column not found. Verify the actual target column.")

    print("\\nAudit complete. Original CSV was not modified.")

if __name__ == "__main__":
    audit_dataset()
