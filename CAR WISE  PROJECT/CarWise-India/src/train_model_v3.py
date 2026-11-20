
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

V1_FILE = DATA_DIR / "carwise_v2_model_data.csv"
V3_FILE = DATA_DIR / "carwise_v3_model_data.csv"

# ==================================================
# 2. LOAD DATA
# ==================================================

v1_df = pd.read_csv(V1_FILE)
v3_df = pd.read_csv(V3_FILE)

if len(v1_df) != len(v3_df):
    raise ValueError("V1 and V3 datasets have different row counts.")

if not v1_df["selling_price"].equals(v3_df["selling_price"]):
    raise ValueError("Target rows are not aligned.")

print("V1 shape:", v1_df.shape)
print("V3 shape:", v3_df.shape)

# ==================================================
# 3. DEFINE FEATURES
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
# 4. GROUP SPLIT
# ==================================================

# Use V1 feature combinations as groups.
# Identical configurations remain in the same split.

group_columns = numerical_features + v1_categorical

groups = pd.util.hash_pandas_object(
    v1_df[group_columns].fillna("MISSING"),
    index=False
).values

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=2026
)

train_idx, test_idx = next(
    splitter.split(v1_df, groups=groups)
)

y = v1_df[target]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining rows:", len(train_idx))
print("Testing rows:", len(test_idx))

# ==================================================
# 5. MODEL TRAINING FUNCTION
# ==================================================

def train_and_evaluate(df, categorical_features, model_name):

    features = numerical_features + categorical_features

    X = df[features]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

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

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessing),
        ("model", model)
    ])

    print(f"\nTraining {model_name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    print(f"\n{model_name} Results")
    print("-" * 40)
    print(f"MAE:  ₹{mae:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R2:   {r2:.4f}")

    return {
        "model_name": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }, pipeline, predictions


# ==================================================
# 6. TRAIN V1, V2 AND V3
# ==================================================

results = []

# V1: Original features
v1_result, v1_model, v1_pred = train_and_evaluate(
    v1_df,
    v1_categorical,
    "V1 Baseline"
)

results.append(v1_result)

# V2: Add car model
v2_result, v2_model, v2_pred = train_and_evaluate(
    v1_df,
    v2_categorical,
    "V2 Car Model"
)

results.append(v2_result)

# V3: Add variant details
v3_result, v3_model, v3_pred = train_and_evaluate(
    v3_df,
    v3_categorical,
    "V3 Model + Variant"
)

results.append(v3_result)

# ==================================================
# 7. COMPARISON
# ==================================================

comparison = pd.DataFrame(results)

print("\n===== FINAL MODEL COMPARISON =====")

print(
    comparison.to_string(index=False)
)

comparison.to_csv(
    ARTIFACT_DIR / "v1_v2_v3_comparison.csv",
    index=False
)

# ==================================================
# 8. SAVE V3 MODEL
# ==================================================

joblib.dump(
    v3_model,
    ARTIFACT_DIR / "v3_model.joblib"
)

# ==================================================
# 9. SAVE PREDICTIONS
# ==================================================

predictions_df = pd.DataFrame({
    "Actual_Price": y_test.values,
    "V1_Predicted": v1_pred,
    "V2_Predicted": v2_pred,
    "V3_Predicted": v3_pred
})

predictions_df.to_csv(
    ARTIFACT_DIR / "v1_v2_v3_predictions.csv",
    index=False
)

print("\nV3 model saved successfully!")
print("Comparison CSV saved!")
print("Predictions CSV saved!")

print("\n===== V3 TRAINING COMPLETED =====")
