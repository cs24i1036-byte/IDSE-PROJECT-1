
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "artifacts" / "final_model.joblib"
DATA_FILE = BASE_DIR / "data" / "processed" / "carwise_model_data.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"

# ==================================================
# 2. CHECK FILES
# ==================================================

if not MODEL_FILE.exists():
    print("ERROR: final_model.joblib not found!")
    raise SystemExit(1)

if not DATA_FILE.exists():
    print("ERROR: carwise_model_data.csv not found!")
    raise SystemExit(1)

# ==================================================
# 3. LOAD MODEL AND DATA
# ==================================================

model = joblib.load(MODEL_FILE)
df = pd.read_csv(DATA_FILE)

target = "selling_price"

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
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

features = numerical_features + categorical_features

X = df[features]
y = df[target]

print("\n" + "=" * 65)
print("       CARWISE INDIA - FEATURE INVESTIGATION")
print("=" * 65)

print("\nDataset shape:", df.shape)

# ==================================================
# 4. EXTRACT PIPELINE COMPONENTS
# ==================================================

if not hasattr(model, "named_steps"):
    print("ERROR: The loaded model is not a pipeline.")
    raise SystemExit(1)

print("\nPipeline steps:")
print(list(model.named_steps.keys()))

preprocessor = model.named_steps.get("preprocessor")

if preprocessor is None:
    print("ERROR: Preprocessor step not found.")
    raise SystemExit(1)

# Find the trained Random Forest estimator
forest = None

for step_name, step in model.named_steps.items():
    if hasattr(step, "feature_importances_"):
        forest = step
        print("\nFinal estimator:", step_name)
        break

if forest is None:
    print("ERROR: Feature importance is unavailable.")
    raise SystemExit(1)

# ==================================================
# 5. FEATURE IMPORTANCE
# ==================================================

feature_names = preprocessor.get_feature_names_out()

importance_values = forest.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== TOP 15 INDIVIDUAL FEATURES =====\n")

print(
    importance_df.head(15).to_string(index=False)
)

# Save individual feature importance
importance_df.to_csv(
    ARTIFACTS_DIR / "investigation_feature_importance.csv",
    index=False
)

# ==================================================
# 6. GROUP IMPORTANCE BY ORIGINAL FEATURE
# ==================================================

def get_original_feature(name):

    # Remove transformer prefixes such as
    # num__ and cat__
    name = name.split("__", 1)[-1]

    # Match original input feature names
    for feature in features:
        if name == feature or name.startswith(feature + "_"):
            return feature

    return name


importance_df["Original_Feature"] = (
    importance_df["Feature"].apply(get_original_feature)
)

grouped_importance = (
    importance_df.groupby("Original_Feature")["Importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

print("\n===== IMPORTANCE BY ORIGINAL FEATURE =====\n")

print(grouped_importance.to_string(index=False))

grouped_importance.to_csv(
    ARTIFACTS_DIR / "grouped_feature_importance.csv",
    index=False
)

# ==================================================
# 7. RECREATE THE FINAL HOLDOUT SPLIT
# ==================================================

groups = pd.util.hash_pandas_object(
    X,
    index=False
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=2026
)

development_indices, holdout_indices = next(
    splitter.split(X, y, groups=groups)
)

X_holdout = X.iloc[holdout_indices]
y_holdout = y.iloc[holdout_indices]

# Generate predictions in the same holdout order
holdout_predictions = model.predict(X_holdout)

holdout_df = X_holdout.copy()

holdout_df["Actual_Price"] = y_holdout.values
holdout_df["Predicted_Price"] = holdout_predictions

holdout_df["Residual"] = (
    holdout_df["Actual_Price"] -
    holdout_df["Predicted_Price"]
)

holdout_df["Absolute_Error"] = (
    holdout_df["Residual"].abs()
)

# ==================================================
# 8. LUXURY CAR INVESTIGATION
# ==================================================

luxury_cars = holdout_df[
    holdout_df["Actual_Price"] >= 3000000
].copy()

print("\n===== LUXURY CAR INVESTIGATION =====")

print("\nTotal luxury cars in holdout:", len(luxury_cars))

if len(luxury_cars) > 0:

    print("\nLuxury car error summary:\n")

    print(
        luxury_cars["Absolute_Error"].describe()
    )

    print("\n===== 10 LARGEST LUXURY CAR ERRORS =====\n")

    columns_to_show = [
        "brand",
        "car_age",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "fuel",
        "transmission",
        "Actual_Price",
        "Predicted_Price",
        "Residual",
        "Absolute_Error"
    ]

    print(
        luxury_cars.nlargest(
            10,
            "Absolute_Error"
        )[columns_to_show].to_string(index=False)
    )

    luxury_cars.to_csv(
        ARTIFACTS_DIR / "luxury_car_investigation.csv",
        index=False
    )

else:
    print("No luxury cars found in the holdout dataset.")

# ==================================================
# 9. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("          INVESTIGATION COMPLETED")
print("=" * 65)

print("\nGenerated files:")

print("1. investigation_feature_importance.csv")
print("2. grouped_feature_importance.csv")
print("3. luxury_car_investigation.csv")

print("\nAll files are saved in the artifacts folder.")
