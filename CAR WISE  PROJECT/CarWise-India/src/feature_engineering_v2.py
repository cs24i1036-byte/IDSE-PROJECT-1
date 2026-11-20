
import pandas as pd
import numpy as np
import re
from pathlib import Path

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "car_details_deduplicated.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "carwise_v2_model_data.csv"
)

# ==================================================
# 2. LOAD DATA
# ==================================================

if not INPUT_FILE.exists():
    print("ERROR: Deduplicated dataset not found!")
    raise SystemExit(1)

df = pd.read_csv(INPUT_FILE)

print("\n===== CARWISE V2 FEATURE ENGINEERING =====")

print("\nOriginal dataset shape:", df.shape)

# Preserve original car name
df["name"] = df["name"].astype("string").str.strip()

# ==================================================
# 3. EXTRACT CAR MODEL
# ==================================================

def extract_model(row):

    name = row["name"]
    brand = row["brand"]

    if pd.isna(name) or pd.isna(brand):
        return "Unknown"

    # Remove brand from beginning of car name
    pattern = r"^" + re.escape(str(brand)) + r"\s+"

    model_name = re.sub(
        pattern,
        "",
        str(name),
        count=1,
        flags=re.IGNORECASE
    ).strip()

    if not model_name:
        return "Unknown"

    # Handle common multi-word model names
    tokens = model_name.split()

    if len(tokens) >= 2:

        first_two = " ".join(tokens[:2]).lower()

        if first_two in [
            "3 series",
            "5 series",
            "6 series",
            "7 series",
            "grand i10",
            "range rover",
            "new swift",
            "land cruiser"
        ]:
            return " ".join(tokens[:2])

    # Default: first token as model family
    return tokens[0]


df["car_model"] = df.apply(
    extract_model,
    axis=1
)

# ==================================================
# 4. FEATURE CLEANING
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

# Treat non-positive specifications as missing
for column in ["mileage", "engine", "max_power"]:

    df.loc[df[column] <= 0, column] = np.nan

# ==================================================
# 5. FEATURE ENGINEERING
# ==================================================

# Dataset is a historical snapshot.
# Using 2020 as the reference year for consistency
# with the original CarWise V1 experiment.

df["car_age"] = 2020 - df["year"]

# ==================================================
# 6. SELECT MODEL FEATURES
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
# 7. DATASET SUMMARY
# ==================================================

print("\n===== MODEL DATASET SUMMARY =====")

print("\nFinal shape:", model_df.shape)

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

print("\nMissing values:")
print(model_df.isnull().sum())

print("\nTop 15 car models:")
print(model_df["car_model"].value_counts().head(15))

print("\nCar model unique values:")
print(model_df["car_model"].nunique())

print("\nSample car name and extracted model:")

print(
    df[["name", "brand", "car_model"]]
    .head(15)
    .to_string(index=False)
)

# ==================================================
# 8. SAVE V2 DATASET
# ==================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

model_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nV2 dataset saved successfully!")
print(OUTPUT_FILE)

print("\n===== FEATURE ENGINEERING COMPLETED =====")
