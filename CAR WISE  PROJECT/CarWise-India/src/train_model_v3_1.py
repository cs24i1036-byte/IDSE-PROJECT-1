
import pandas as pd
import numpy as np
import joblib

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

V2_FILE = DATA_DIR / "carwise_v2_model_data.csv"
V3_FILE = DATA_DIR / "carwise_v3_model_data.csv"
V31_FILE = DATA_DIR / "carwise_v3_1_model_data.csv"


# ==================================================
# 2. LOAD DATA
# ==================================================

v2_df = pd.read_csv(V2_FILE)
v3_df = pd.read_csv(V3_FILE)
v31_df = pd.read_csv(V31_FILE)

datasets = [v2_df, v3_df, v31_df]

for df in datasets:

    if len(df) != len(v2_df):
        raise ValueError("Dataset row counts do not match.")

    if not df["selling_price"].equals(v2_df["selling_price"]):
        raise ValueError("Target rows are not aligned.")


# ==================================================
# 3. FEATURES
# ==================================================

numerical_features = [
    "car_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

v2_categorical = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "car_model"
]

v3_categorical = v2_categorical + [
    "variant_details"
]

v31_categorical = v3_categorical


# ==================================================
# 4. GROUPS AND FIVE SPLITS
# ==================================================

group_columns = numerical_features + [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

groups = pd.util.hash_pandas_object(
    v2_df[group_columns].fillna("MISSING"),
    index=False
).values

X_reference = v2_df[numerical_features + [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]]

y = v2_df["selling_price"]

splitter = GroupShuffleSplit(
    n_splits=5,
    test_size=0.20,
    random_state=42
)

splits = list(
    splitter.split(X_reference, y, groups=groups)
)


# ==================================================
# 5. MODEL BUILDER
# ==================================================

def build_model(categorical_features):

    numerical_pipeline = Pipeline([
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
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocessor", preprocessing),
        ("model", rf)
    ])


# ==================================================
# 6. EXPERIMENTS
# ==================================================

experiments = {
    "V2 Car Model": (v2_df, v2_categorical),
    "V3 Old Extraction": (v3_df, v3_categorical),
    "V3.1 Corrected Extraction": (v31_df, v31_categorical)
}

results = []

for split_number, (train_idx, test_idx) in enumerate(splits, start=1):

    print("\n" + "=" * 60)
    print(f"VALIDATION SPLIT {split_number}")
    print("=" * 60)

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    for model_name, (df, categorical_features) in experiments.items():

        features = numerical_features + categorical_features

        X_train = df.iloc[train_idx][features]
        X_test = df.iloc[test_idx][features]

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

        results.append({
            "Split": split_number,
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })


# ==================================================
# 7. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    ARTIFACT_DIR / "v3_1_validation_results.csv",
    index=False
)


# ==================================================
# 8. SUMMARY
# ==================================================

summary = results_df.groupby("Model").agg(
    MAE_Mean=("MAE", "mean"),
    MAE_Std=("MAE", "std"),
    RMSE_Mean=("RMSE", "mean"),
    RMSE_Std=("RMSE", "std"),
    R2_Mean=("R2", "mean"),
    R2_Std=("R2", "std")
).reset_index()

summary.to_csv(
    ARTIFACT_DIR / "v3_1_validation_summary.csv",
    index=False
)

print("\n===== FINAL V2 vs V3 vs V3.1 COMPARISON =====")

print(summary.to_string(index=False))

print("\nResults saved successfully!")

print("\nV3.1 TRAINING EXPERIMENT COMPLETED!")
