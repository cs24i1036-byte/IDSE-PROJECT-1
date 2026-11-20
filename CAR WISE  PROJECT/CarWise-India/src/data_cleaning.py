
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "raw" / "car_details_v3.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "car_details_cleaned.csv"


# --------------------------------------------------
# 2. LOAD ORIGINAL DATA
# --------------------------------------------------

df = pd.read_csv(RAW_FILE)

print("\n========== CARWISE DATA CLEANING ==========")
print("Original shape:", df.shape)


# --------------------------------------------------
# 3. CREATE A COPY
# --------------------------------------------------

clean_df = df.copy()


# --------------------------------------------------
# 4. CONVERT MEASUREMENT COLUMNS TO NUMERIC
# --------------------------------------------------

measurement_columns = [
    "mileage",
    "engine",
    "max_power"
]

for column in measurement_columns:

    clean_df[column] = pd.to_numeric(
        clean_df[column]
        .astype("string")
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False),
        errors="coerce"
    )


# Convert seats to numeric as well
clean_df["seats"] = pd.to_numeric(
    clean_df["seats"],
    errors="coerce"
)


# --------------------------------------------------
# 5. EXTRACT BRAND FROM CAR NAME
# --------------------------------------------------

# Handle known multi-word brand names first.
multi_word_brands = [
    "Land Rover",
    "Mercedes-Benz"
]

def extract_brand(car_name):

    if pd.isna(car_name):
        return pd.NA

    car_name = str(car_name).strip()

    for brand in multi_word_brands:

        if car_name.lower().startswith(brand.lower()):
            return brand

    # Initial extraction for other brands.
    return car_name.split()[0]


clean_df["brand"] = clean_df["name"].apply(extract_brand)


# --------------------------------------------------
# 6. CHECK DUPLICATES
# --------------------------------------------------

duplicate_count = clean_df.duplicated().sum()

print("\nExact duplicate rows:", duplicate_count)

# Important:
# We are NOT deleting duplicates at this stage.


# --------------------------------------------------
# 7. CHECK MISSING VALUES
# --------------------------------------------------

print("\nMissing values after conversion:")

print(
    clean_df.isnull()
    .sum()
    .sort_values(ascending=False)
)


# --------------------------------------------------
# 8. CHECK DATA TYPES
# --------------------------------------------------

print("\nData types after cleaning:")

print(clean_df.dtypes)


# --------------------------------------------------
# 9. BASIC VALIDATION
# --------------------------------------------------

print("\nNumeric column summary:")

numeric_columns = [
    "year",
    "selling_price",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

print(clean_df[numeric_columns].describe())


print("\nUnique fuel values:")
print(clean_df["fuel"].unique())

print("\nUnique transmission values:")
print(clean_df["transmission"].unique())

print("\nUnique owner values:")
print(clean_df["owner"].unique())

print("\nExtracted brands:")
print(clean_df["brand"].value_counts().head(15))



# --------------------------------------------------
# 9.5 INVESTIGATE SUSPICIOUS VALUES
# --------------------------------------------------

print("\n========== DATA QUALITY INVESTIGATION ==========")

# A. Find rows where max_power could not be converted
print("\n1. Rows with missing max_power:")

print(
    df.loc[
        clean_df["max_power"].isna() & df["max_power"].notna(),
        ["name", "max_power"]
    ].to_string(index=False)
)


# B. Find zero mileage
print("\n2. Rows with zero mileage:")

print(
    clean_df.loc[
        clean_df["mileage"] == 0,
        ["name", "mileage", "selling_price"]
    ].to_string(index=False)
)


# C. Find zero max_power
print("\n3. Rows with zero max_power:")

print(
    clean_df.loc[
        clean_df["max_power"] == 0,
        ["name", "max_power", "selling_price"]
    ].to_string(index=False)
)


# D. Check brand extraction for less frequent brands
print("\n4. All extracted brands:")

print(clean_df["brand"].value_counts().to_string())


# E. Check suspicious values in numerical columns
print("\n5. Records with unusually high kilometres:")

print(
    clean_df.loc[
        clean_df["km_driven"] > 500000,
        ["name", "year", "km_driven", "selling_price"]
    ].to_string(index=False)
)


# --------------------------------------------------
# 10. SAVE CLEANED DATA
# --------------------------------------------------

clean_df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaned dataset saved successfully!")
print("Output path:", OUTPUT_FILE)
print("Final shape:", clean_df.shape)

print("\nOriginal dataset remains unchanged.")
