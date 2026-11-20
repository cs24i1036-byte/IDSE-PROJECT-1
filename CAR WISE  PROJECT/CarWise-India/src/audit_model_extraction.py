
import pandas as pd
from pathlib import Path

# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

RAW_FILE = DATA_DIR / "car_details_deduplicated.csv"
V31_FILE = DATA_DIR / "carwise_v3_1_model_data.csv"

OUTPUT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. LOAD DATA
# ==================================================

raw = pd.read_csv(RAW_FILE)
v31 = pd.read_csv(V31_FILE)

if len(raw) != len(v31):
    raise ValueError("Raw and V3.1 datasets have different row counts.")

df = raw[["name", "brand"]].copy()

df["car_model"] = v31["car_model"]
df["variant_details"] = v31["variant_details"]

print("\n===== MODEL EXTRACTION QUALITY AUDIT =====")

print("Total records:", len(df))
print("Unique brands:", df["brand"].nunique())
print("Unique models:", df["car_model"].nunique())
print("Unique variants:", df["variant_details"].nunique())

# ==================================================
# 3. UNSPECIFIED VARIANT AUDIT
# ==================================================

unspecified = df[
    df["variant_details"] == "Unspecified"
].copy()

print("\n===== UNSPECIFIED VARIANT AUDIT =====")

print("Unspecified records:", len(unspecified))

if len(df) > 0:
    print(
        "Unspecified percentage:",
        round(len(unspecified) / len(df) * 100, 2),
        "%"
    )

print("\nUnspecified records by brand:")

print(
    unspecified["brand"]
    .value_counts()
    .to_string()
)

print("\nSample unspecified records:")

print(
    unspecified[
        ["name", "brand", "car_model"]
    ].head(50).to_string(index=False)
)

# ==================================================
# 4. RARE MODEL AUDIT
# ==================================================

model_counts = df["car_model"].value_counts()

rare_models = model_counts[
    model_counts <= 3
]

print("\n===== RARE MODEL AUDIT =====")

print("Models occurring 3 times or fewer:", len(rare_models))

print("\nRare model examples:")

print(rare_models.head(60).to_string())

# ==================================================
# 5. BRAND-WISE MODEL DISTRIBUTION
# ==================================================

print("\n===== BRAND-WISE MODEL COUNTS =====")

brand_model_counts = (
    df.groupby("brand")["car_model"]
    .nunique()
    .sort_values(ascending=False)
)

print(brand_model_counts.to_string())

# ==================================================
# 6. CHECK IMPORTANT MODEL FAMILIES
# ==================================================

print("\n===== IMPORTANT MODEL FAMILY CHECK =====")

checks = [
    "Alto 800",
    "Alto",
    "Alto K10",
    "Swift Dzire",
    "Swift",
    "Wagon R",
    "Grand i10",
    "3 Series",
    "5 Series",
    "Range Rover Evoque",
    "Land Cruiser Prado"
]

for model in checks:

    matching = df[
        df["car_model"].str.lower() == model.lower()
    ]

    print(f"\n{model}: {len(matching)} records")

    if not matching.empty:
        print(
            matching[
                ["name", "car_model", "variant_details"]
            ].head(5).to_string(index=False)
        )

# ==================================================
# 7. SUSPICIOUS MODEL NAMES
# ==================================================

# These are candidates for manual inspection,
# not automatically incorrect model names.

suspicious = df[
    (df["car_model"].str.contains(
        r"\b(diesel|petrol|vdi|vxi|lxi|bsiii|bsiv|tdi|crdi)\b",
        case=False,
        na=False
    ))
    |
    (df["variant_details"] == "Unspecified")
].copy()

print("\n===== SUSPICIOUS EXTRACTION CANDIDATES =====")

print("Candidate records:", len(suspicious))

print(
    suspicious[
        ["name", "brand", "car_model", "variant_details"]
    ].head(60).to_string(index=False)
)

# ==================================================
# 8. SAVE AUDIT FILES
# ==================================================

unspecified.to_csv(
    OUTPUT_DIR / "unspecified_variant_audit.csv",
    index=False
)

rare_models.rename("record_count").to_csv(
    OUTPUT_DIR / "rare_models_audit.csv"
)

suspicious.to_csv(
    OUTPUT_DIR / "suspicious_extractions_audit.csv",
    index=False
)

print("\nAudit files saved in artifacts/")

print("\nMODEL EXTRACTION AUDIT COMPLETED!")
