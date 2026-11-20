
import pandas as pd
import re
from pathlib import Path

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CLEAN_FILE = (
    BASE_DIR / "data" / "processed" /
    "car_details_deduplicated.csv"
)

V2_FILE = (
    BASE_DIR / "data" / "processed" /
    "carwise_v2_model_data.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "carwise_v3_model_data.csv"
)

# ==================================================
# 2. LOAD DATA
# ==================================================

if not CLEAN_FILE.exists() or not V2_FILE.exists():
    print("ERROR: Required input files not found!")
    raise SystemExit(1)

clean = pd.read_csv(CLEAN_FILE)
v2 = pd.read_csv(V2_FILE)

# Verify row alignment before combining information
if len(clean) != len(v2):
    raise ValueError("Clean and V2 datasets have different row counts.")

if not clean["selling_price"].equals(v2["selling_price"]):
    raise ValueError("Dataset rows are not aligned. Stop before combining.")

df = clean.copy()

# Reuse the V2 car_model feature
df["car_model"] = v2["car_model"].values

print("\n===== CARWISE V3 FEATURE ENGINEERING =====")
print("\nInput dataset shape:", df.shape)

# ==================================================
# 3. EXTRACT VARIANT DETAILS
# ==================================================

def extract_variant(row):

    name = str(row["name"]).strip()
    brand = str(row["brand"]).strip()
    model = str(row["car_model"]).strip()

    # Remove brand from the beginning
    brand_pattern = r"^" + re.escape(brand) + r"\s+"

    remaining = re.sub(
        brand_pattern,
        "",
        name,
        count=1,
        flags=re.IGNORECASE
    ).strip()

    # Remove extracted model from the beginning
    model_pattern = r"^" + re.escape(model) + r"(?:\s+|$)"

    remaining = re.sub(
        model_pattern,
        "",
        remaining,
        count=1,
        flags=re.IGNORECASE
    ).strip()

    # Clean repeated spaces
    remaining = re.sub(r"\s+", " ", remaining)

    if not remaining:
        return "Unspecified"

    return remaining


df["variant_details"] = df.apply(
    extract_variant,
    axis=1
)

# ==================================================
# 4. NUMERICAL CLEANING
# ==================================================

numeric_columns = [
    "mileage",
    "engine",
    "max_power",
    "seats"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Invalid specifications become missing values
for column in ["mileage", "engine", "max_power"]:

    df.loc[df[column] <= 0, column] = pd.NA

    # ==================================================
# 4. CREATE CAR AGE
# ==================================================

df["car_age"] = 2020 - df["year"]

# ==================================================
# 5. FEATURE SELECTION
# ==================================================

numerical_features = [
    "car_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

categorical_features = [
    "brand",
    "car_model",
    "variant_details",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

target = "selling_price"

selected_columns = (
    numerical_features +
    categorical_features +
    [target]
)

model_df = df[selected_columns].copy()

# ==================================================
# 6. DATASET SUMMARY
# ==================================================

print("\n===== V3 DATASET SUMMARY =====")

print("\nFinal dataset shape:", model_df.shape)

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

print("\nMissing values:")
print(model_df.isnull().sum())

print("\nUnique car models:")
print(model_df["car_model"].nunique())

print("\nUnique variant descriptions:")
print(model_df["variant_details"].nunique())

print("\nTop 20 variant descriptions:")
print(
    model_df["variant_details"]
    .value_counts()
    .head(20)
)

# ==================================================
# 7. INSPECT EXTRACTION QUALITY
# ==================================================

print("\n===== SAMPLE MODEL AND VARIANT EXTRACTION =====")

print(
    df[
        ["name", "brand", "car_model", "variant_details"]
    ].head(25).to_string(index=False)
)

print("\n===== MOST FREQUENT MODEL-VARIANT COMBINATIONS =====")

print(
    model_df.groupby(
        ["car_model", "variant_details"]
    ).size()
    .sort_values(ascending=False)
    .head(15)
    .to_string()
)

# ==================================================
# 8. SAVE V3 DATASET
# ==================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

model_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nV3 dataset saved successfully!")
print(OUTPUT_FILE)

print("\n===== V3 FEATURE ENGINEERING COMPLETED =====")
