
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "carwise_model_data.csv"
)

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = ARTIFACTS_DIR / "model_results.csv"
BEST_MODEL_FILE = ARTIFACTS_DIR / "best_model.joblib"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("\n========== CARWISE MODEL TRAINING ==========")

print("\nDataset shape:", df.shape)


# --------------------------------------------------
# 3. DEFINE FEATURES AND TARGET
# --------------------------------------------------

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

X = df[numerical_features + categorical_features].copy()

y = df[target].copy()


# --------------------------------------------------
# 4. GROUP-AWARE TRAIN-TEST SPLIT
# --------------------------------------------------

# Identical feature combinations stay in the same split.
groups = pd.util.hash_pandas_object(
    X,
    index=False
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Training percentage:",
    round(len(X_train) / len(X) * 100, 2),
    "%"
)

print(
    "Testing percentage:",
    round(len(X_test) / len(X) * 100, 2),
    "%"
)


# --------------------------------------------------
# 5. BUILD PREPROCESSING PIPELINE
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 6. DEFINE MACHINE LEARNING MODELS
# --------------------------------------------------

models = {

    "Dummy Regressor": DummyRegressor(
        strategy="mean"
    ),

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 7. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

results = []
trained_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 55)
    print("Training:", model_name)

    # Each model receives its own independent pipeline.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    print("MAE :", round(mae, 2))
    print("MSE :", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[model_name] = pipeline


# --------------------------------------------------
# 8. CREATE MODEL COMPARISON TABLE
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)

print("\n========== MODEL COMPARISON ==========")

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:,.2f}".format,
            "MSE": "{:,.2f}".format,
            "RMSE": "{:,.2f}".format,
            "R2": "{:.4f}".format
        }
    )
)


# --------------------------------------------------
# 9. SELECT BEST MODEL
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest model based on RMSE:", best_model_name)


# --------------------------------------------------
# 10. SAVE RESULTS AND BEST MODEL
# --------------------------------------------------

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

joblib.dump(
    best_model,
    BEST_MODEL_FILE
)

print("\nModel results saved:", RESULTS_FILE)

print("Best model saved:", BEST_MODEL_FILE)

print("\nModel training completed successfully!")
