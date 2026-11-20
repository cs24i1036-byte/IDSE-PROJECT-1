
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "car_details_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "carwise_model_data.csv"


# --------------------------------------------------
# 2. LOAD CLEANED DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\n========== CARWISE FEATURE ENGINEERING ==========")

print("\nOriginal dataset shape:", df.shape)


# --------------------------------------------------
# 3. HANDLE INVALID NUMERICAL VALUES
# --------------------------------------------------

positive_columns = [
    "mileage",
    "engine",
    "max_power"
]

for column in positive_columns:

    df.loc[df[column] <= 0, column] = pd.NA


print("\nMissing values after invalid-value checks:")

print(df[positive_columns].isna().sum())


# --------------------------------------------------
# 4. CREATE CAR AGE
# --------------------------------------------------

REFERENCE_YEAR = 2020

df["car_age"] = REFERENCE_YEAR - df["year"]

print("\nCar age created successfully.")

print(df["car_age"].describe())


# --------------------------------------------------
# 5. SELECT NUMERICAL FEATURES
# --------------------------------------------------

numerical_features = [
    "car_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]


# --------------------------------------------------
# 6. SELECT CATEGORICAL FEATURES
# --------------------------------------------------

categorical_features = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


# --------------------------------------------------
# 7. SELECT TARGET
# --------------------------------------------------

target = "selling_price"


# --------------------------------------------------
# 8. CREATE MODEL DATASET
# --------------------------------------------------

selected_columns = (
    numerical_features
    + categorical_features
    + [target]
)

model_df = df[selected_columns].copy()


# --------------------------------------------------
# 9. SEPARATE X AND Y
# --------------------------------------------------

X = model_df.drop(columns=[target])

y = model_df[target]


print("\nFeature matrix X shape:", X.shape)

print("Target vector y shape:", y.shape)


# --------------------------------------------------
# 10. DISPLAY FEATURE INFORMATION
# --------------------------------------------------

print("\nNumerical features:")

print(numerical_features)

print("\nCategorical features:")

print(categorical_features)

print("\nTarget:")

print(target)


# --------------------------------------------------
# 11. CHECK MISSING VALUES
# --------------------------------------------------

print("\nMissing values in model dataset:")

print(
    model_df.isna()
    .sum()
    .sort_values(ascending=False)
)


# --------------------------------------------------
# 12. CHECK DUPLICATES
# --------------------------------------------------

print("\nDuplicate rows in model dataset:")

print(model_df.duplicated().sum())

# Keep duplicates for now.
# Their impact will be investigated separately.


# --------------------------------------------------
# 13. SAVE MODEL DATASET
# --------------------------------------------------

model_df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature dataset saved successfully!")

print("Location:", OUTPUT_FILE)

print("\nFinal model dataset shape:", model_df.shape)

print("\nFeature engineering completed successfully!")
