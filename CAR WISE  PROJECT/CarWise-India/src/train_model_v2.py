
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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR / "data" / "processed" /
    "carwise_v2_model_data.csv"
)

ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. LOAD DATA
# ==================================================

df = pd.read_csv(DATA_FILE)

target = "selling_price"

# V1 features: without car_model
v1_numerical = [
    "car_age", "km_driven", "mileage",
    "engine", "max_power", "seats"
]

v1_categorical = [
    "brand", "fuel", "seller_type",
    "transmission", "owner"
]

# V2 features: with car_model
v2_numerical = v1_numerical.copy()

v2_categorical = [
    "brand", "car_model", "fuel",
    "seller_type", "transmission", "owner"
]

v1_features = v1_numerical + v1_categorical
v2_features = v2_numerical + v2_categorical

X_v1 = df[v1_features]
X_v2 = df[v2_features]

y = df[target]

print("\n" + "=" * 65)
print("           CARWISE V1 vs V2 MODEL TRAINING")
print("=" * 65)

print("\nDataset shape:", df.shape)

# ==================================================
# 3. CREATE COMMON GROUPED HOLDOUT SPLIT
# ==================================================

# Group using V1 input combinations.
# This keeps identical V1 feature combinations
# together in either development or holdout.

groups = pd.util.hash_pandas_object(
    X_v1,
    index=False
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=2026
)

development_indices, holdout_indices = next(
    splitter.split(X_v1, y, groups=groups)
)

print("\nDevelopment records:", len(development_indices))
print("Holdout records:", len(holdout_indices))

# Same indices for both models
X1_train = X_v1.iloc[development_indices]
X1_test = X_v1.iloc[holdout_indices]

X2_train = X_v2.iloc[development_indices]
X2_test = X_v2.iloc[holdout_indices]

y_train = y.iloc[development_indices]
y_test = y.iloc[holdout_indices]

# ==================================================
# 4. PREPROCESSING FUNCTION
# ==================================================

def create_preprocessor(numerical, categorical):

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical),
        ("categorical", categorical_pipeline, categorical)
    ])

    return preprocessor


# ==================================================
# 5. MODEL CREATION FUNCTION
# ==================================================

def create_model(numerical, categorical):

    pipeline = Pipeline([
        (
            "preprocessor",
            create_preprocessor(numerical, categorical)
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    return pipeline


# ==================================================
# 6. TRAIN V1 BASELINE
# ==================================================

print("\n===== TRAINING V1 BASELINE =====")

model_v1 = create_model(
    v1_numerical,
    v1_categorical
)

model_v1.fit(X1_train, y_train)

pred_v1 = model_v1.predict(X1_test)

print("V1 training completed.")

# ==================================================
# 7. TRAIN V2 IMPROVED MODEL
# ==================================================

print("\n===== TRAINING V2 MODEL =====")

model_v2 = create_model(
    v2_numerical,
    v2_categorical
)

model_v2.fit(X2_train, y_train)

pred_v2 = model_v2.predict(X2_test)

print("V2 training completed.")

# ==================================================
# 8. EVALUATION FUNCTION
# ==================================================

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual, predicted)

    rmse = np.sqrt(mse)

    r2 = r2_score(actual, predicted)

    return {
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


results = []

results.append(
    evaluate_model("V1 Baseline", y_test, pred_v1)
)

results.append(
    evaluate_model("V2 With Car Model", y_test, pred_v2)
)

results_df = pd.DataFrame(results)

# ==================================================
# 9. DISPLAY COMPARISON
# ==================================================

print("\n" + "=" * 75)
print("                 V1 vs V2 PERFORMANCE")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": lambda x: f"₹{x:,.2f}",
            "MSE": lambda x: f"{x:,.2f}",
            "RMSE": lambda x: f"₹{x:,.2f}",
            "R2": lambda x: f"{x:.4f}"
        }
    )
)

# ==================================================
# 10. IMPROVEMENT CALCULATION
# ==================================================

v1_mae = results_df.loc[0, "MAE"]
v2_mae = results_df.loc[1, "MAE"]

v1_rmse = results_df.loc[0, "RMSE"]
v2_rmse = results_df.loc[1, "RMSE"]

print("\n===== PERFORMANCE CHANGE =====")

mae_change = ((v1_mae - v2_mae) / v1_mae) * 100

rmse_change = ((v1_rmse - v2_rmse) / v1_rmse) * 100

print(f"MAE improvement: {mae_change:.2f}%")
print(f"RMSE improvement: {rmse_change:.2f}%")

if v2_rmse < v1_rmse:
    print("\nV2 has lower RMSE than V1.")
else:
    print("\nV2 did not improve RMSE over V1.")

# ==================================================
# 11. SAVE RESULTS AND MODELS
# ==================================================

results_df.to_csv(
    ARTIFACTS_DIR / "v1_vs_v2_comparison.csv",
    index=False
)

joblib.dump(
    model_v1,
    ARTIFACTS_DIR / "v1_deduplicated_baseline.joblib"
)

joblib.dump(
    model_v2,
    ARTIFACTS_DIR / "v2_model.joblib"
)

# Save V2 holdout predictions
predictions = pd.DataFrame({
    "Actual_Price": y_test.values,
    "V1_Predicted_Price": pred_v1,
    "V2_Predicted_Price": pred_v2
})

predictions.to_csv(
    ARTIFACTS_DIR / "v1_vs_v2_predictions.csv",
    index=False
)

print("\n===== FILES SAVED =====")

print("1. v1_vs_v2_comparison.csv")
print("2. v1_deduplicated_baseline.joblib")
print("3. v2_model.joblib")
print("4. v1_vs_v2_predictions.csv")

print("\n===== TRAINING COMPLETED =====")
