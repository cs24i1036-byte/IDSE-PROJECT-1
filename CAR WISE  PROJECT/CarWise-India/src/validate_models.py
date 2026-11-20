
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
ARTIFACT_DIR = BASE_DIR / "artifacts"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

V1_FILE = DATA_DIR / "carwise_v2_model_data.csv"
V3_FILE = DATA_DIR / "carwise_v3_model_data.csv"


# ==================================================
# 2. LOAD DATA
# ==================================================

v1_df = pd.read_csv(V1_FILE)
v3_df = pd.read_csv(V3_FILE)

if len(v1_df) != len(v3_df):
    raise ValueError("V1 and V3 row counts do not match.")

if not v1_df["selling_price"].equals(v3_df["selling_price"]):
    raise ValueError("Target columns are not aligned.")

print("Dataset loaded successfully.")
print("Rows:", len(v1_df))


# ==================================================
# 3. FEATURE DEFINITIONS
# ==================================================

numerical_features = [
    "car_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

v1_categorical = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

v2_categorical = v1_categorical + [
    "car_model"
]

v3_categorical = v2_categorical + [
    "variant_details"
]

target = "selling_price"


# ==================================================
# 4. GROUP CREATION
# ==================================================

group_columns = numerical_features + v1_categorical

groups = pd.util.hash_pandas_object(
    v1_df[group_columns].fillna("MISSING"),
    index=False
).values

X_reference = v1_df[numerical_features + v1_categorical]
y = v1_df[target]


# ==================================================
# 5. CREATE FIVE GROUPED SPLITS
# ==================================================

splitter = GroupShuffleSplit(
    n_splits=5,
    test_size=0.20,
    random_state=42
)

splits = list(
    splitter.split(X_reference, y, groups=groups)
)

print("\nNumber of validation splits:", len(splits))


# ==================================================
# 6. MODEL TRAINING FUNCTION
# ==================================================

def build_model(categorical_features):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=10
        ))
    ])

    preprocessing = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    random_forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocessor", preprocessing),
        ("model", random_forest)
    ])


# ==================================================
# 7. RUN VALIDATION
# ==================================================

experiments = {
    "V1 Baseline": (v1_df, v1_categorical),
    "V2 Car Model": (v1_df, v2_categorical),
    "V3 Model + Variant": (v3_df, v3_categorical)
}

all_results = []

for split_number, (train_idx, test_idx) in enumerate(splits, start=1):

    print("\n" + "=" * 55)
    print(f"VALIDATION SPLIT {split_number}")
    print("=" * 55)

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("Training rows:", len(train_idx))
    print("Testing rows:", len(test_idx))

    for model_name, (df, categorical_features) in experiments.items():

        features = numerical_features + categorical_features

        X = df[features]

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        model = build_model(categorical_features)

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        mse = mean_squared_error(y_test, predictions)

        rmse = np.sqrt(mse)

        r2 = r2_score(y_test, predictions)

        print(f"MAE:  ₹{mae:,.2f}")
        print(f"RMSE: ₹{rmse:,.2f}")
        print(f"R2:   {r2:.4f}")

        all_results.append({
            "Split": split_number,
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })


# ==================================================
# 8. SAVE INDIVIDUAL SPLIT RESULTS
# ==================================================

results_df = pd.DataFrame(all_results)

results_file = ARTIFACT_DIR / "five_split_validation_results.csv"

results_df.to_csv(results_file, index=False)


# ==================================================
# 9. CALCULATE MEAN AND STANDARD DEVIATION
# ==================================================

summary = results_df.groupby("Model").agg(
    MAE_Mean=("MAE", "mean"),
    MAE_Std=("MAE", "std"),

    RMSE_Mean=("RMSE", "mean"),
    RMSE_Std=("RMSE", "std"),

    R2_Mean=("R2", "mean"),
    R2_Std=("R2", "std")
).reset_index()

summary_file = ARTIFACT_DIR / "five_split_validation_summary.csv"

summary.to_csv(summary_file, index=False)


# ==================================================
# 10. DISPLAY FINAL COMPARISON
# ==================================================

print("\n" + "=" * 65)
print("FINAL FIVE-SPLIT VALIDATION COMPARISON")
print("=" * 65)

print(summary.to_string(index=False))

print("\nInterpretation:")
print("- Lower MAE is better.")
print("- Lower RMSE is better.")
print("- Higher R2 is better.")
print("- Lower standard deviation indicates more stable performance.")

print("\nResults saved:")
print(results_file)
print(summary_file)

print("\nFIVE-SPLIT VALIDATION COMPLETED!")
