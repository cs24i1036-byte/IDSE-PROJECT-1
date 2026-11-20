
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


# ==========================================
# 1. LOAD V3.1 DATASET
# ==========================================

DATA_PATH = "data/processed/carwise_v3_1_model_data.csv"
OUTPUT_PATH = "artifacts/v3_1_random_forest_final.joblib"

df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("CARWISE V3.1 FINAL MODEL TRAINING")
print("=" * 55)

print("Dataset shape:", df.shape)


# ==========================================
# 2. DEFINE FEATURES AND TARGET
# ==========================================

numeric_features = [
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
    "variant_details",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

target = "selling_price"

X = df[numeric_features + categorical_features]
y = df[target]


# ==========================================
# 3. PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=10
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==========================================
# 4. FINAL RANDOM FOREST
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)

final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf_model)
])


# ==========================================
# 5. TRAIN ON COMPLETE DATASET
# ==========================================

print("\nTraining Random Forest on all records...")

final_pipeline.fit(X, y)

print("Training completed!")


# ==========================================
# 6. SAVE FINAL MODEL
# ==========================================

os.makedirs("artifacts", exist_ok=True)

joblib.dump(final_pipeline, OUTPUT_PATH)

print("\nFinal model saved successfully!")
print("Location:", os.path.abspath(OUTPUT_PATH))

print("\nTraining records:", len(X))
print("Input features:", X.shape[1])
print("Algorithm: Random Forest")
print("Primary selection metric: MAE")

print("\n" + "=" * 55)
print("FINAL MODEL READY")
print("=" * 55)
