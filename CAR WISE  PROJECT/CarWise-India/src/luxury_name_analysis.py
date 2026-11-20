
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent.parent

CLEAN_FILE = BASE_DIR / "data" / "processed" / "car_details_cleaned.csv"
MODEL_FILE = BASE_DIR / "data" / "processed" / "carwise_model_data.csv"

clean = pd.read_csv(CLEAN_FILE)
model_data = pd.read_csv(MODEL_FILE)

# Check that both files have matching row order and target values
if len(clean) != len(model_data):
    raise ValueError("Dataset row counts do not match.")

if not clean["selling_price"].equals(model_data["selling_price"]):
    raise ValueError(
        "Dataset rows are not aligned. Cannot safely recover car names."
    )

features = [
    "car_age", "km_driven", "mileage",
    "engine", "max_power", "seats",
    "brand", "fuel", "seller_type",
    "transmission", "owner"
]

X = model_data[features]
y = model_data["selling_price"]

groups = pd.util.hash_pandas_object(X, index=False)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=2026
)

_, holdout_indices = next(
    splitter.split(X, y, groups=groups)
)

# Recover original names using the same row indices
holdout = clean.iloc[holdout_indices].copy()

holdout["Predicted_Price"] = pd.read_csv(
    BASE_DIR / "artifacts" / "final_predictions.csv"
)["Predicted_Price"].values

holdout["Absolute_Error"] = (
    holdout["selling_price"] - holdout["Predicted_Price"]
).abs()

# Luxury cars: actual price >= 30 lakh
luxury = holdout[
    holdout["selling_price"] >= 3000000
].copy()

print("\n===== LUXURY CAR NAME INVESTIGATION =====\n")

print("Total luxury records:", len(luxury))

print("\n===== 15 LARGEST LUXURY CAR ERRORS =====\n")

columns = [
    "name",
    "brand",
    "year",
    "selling_price",
    "Predicted_Price",
    "Absolute_Error"
]

print(
    luxury.nlargest(15, "Absolute_Error")[columns]
    .to_string(index=False)
)

print("\n===== MOST FREQUENT LUXURY CAR NAMES =====\n")

print(luxury["name"].value_counts().head(15))

output = BASE_DIR / "artifacts" / "luxury_name_analysis.csv"
luxury.to_csv(output, index=False)

print("\nSaved:", output)
