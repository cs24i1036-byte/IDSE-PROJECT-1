
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
    raise ValueError("Dataset row counts do not match.")

if not v1_df["selling_price"].equals(v3_df["selling_price"]):
    raise ValueError("Target rows are not aligned.")

print("Datasets loaded:", len(v1_df))


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

v1_categorical = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

v2_categorical = v1_categorical + ["car_model"]

v3_categorical = v2_categorical + ["variant_details"]

target = "selling_price"


# ==================================================
# 4. PRICE SEGMENTS
# ==================================================

def get_segment(price):

    if price < 500000:
        return "Budget"

    elif price < 1500000:
        return "Mid-range"

    elif price < 3000000:
        return "Premium"

    else:
        return "Luxury"


# ==================================================
# 5. GROUPED SPLITS
# ==================================================

group_columns = numerical_features + v1_categorical

groups = pd.util.hash_pandas_object(
    v1_df[group_columns].fillna("MISSING"),
    index=False
).values

X_reference = v1_df[numerical_features + v1_categorical]
y = v1_df[target]

splitter = GroupShuffleSplit(
    n_splits=5,
    test_size=0.20,
    random_state=42
)

splits = list(
    splitter.split(X_reference, y, groups=groups)
)


# ==================================================
# 6. MODEL BUILDER
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
# 7. EXPERIMENT DEFINITIONS
# ==================================================

experiments = {
    "V1 Baseline": (v1_df, v1_categorical),
    "V2 Car Model": (v1_df, v2_categorical),
    "V3 Model + Variant": (v3_df, v3_categorical)
}

all_results = []


# ==================================================
# 8. TRAIN AND EVALUATE EACH SPLIT
# ==================================================

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

        evaluation_df = pd.DataFrame({
            "Actual_Price": y_test.values,
            "Predicted_Price": predictions
        })

        evaluation_df["Segment"] = (
            evaluation_df["Actual_Price"].apply(get_segment)
        )

        # Evaluate each price segment separately
        for segment in [
            "Budget",
            "Mid-range",
            "Premium",
            "Luxury"
        ]:

            segment_df = evaluation_df[
                evaluation_df["Segment"] == segment
            ]

            actual = segment_df["Actual_Price"]
            predicted = segment_df["Predicted_Price"]

            count = len(segment_df)

            if count == 0:
                continue

            mae = mean_absolute_error(actual, predicted)

            mse = mean_squared_error(actual, predicted)

            rmse = np.sqrt(mse)

            # R2 is undefined for fewer than two samples
            r2 = (
                r2_score(actual, predicted)
                if count >= 2
                else np.nan
            )

            print(
                f"{segment:12} | "
                f"Count: {count:4} | "
                f"MAE: ₹{mae:,.0f} | "
                f"RMSE: ₹{rmse:,.0f} | "
                f"R2: {r2:.4f}"
            )

            all_results.append({
                "Split": split_number,
                "Model": model_name,
                "Segment": segment,
                "Count": count,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })


# ==================================================
# 9. SAVE INDIVIDUAL RESULTS
# ==================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    ARTIFACT_DIR / "segment_validation_results.csv",
    index=False
)


# ==================================================
# 10. SUMMARY: MEAN AND STANDARD DEVIATION
# ==================================================

summary = results_df.groupby(
    ["Model", "Segment"]
).agg(
    Mean_Count=("Count", "mean"),
    MAE_Mean=("MAE", "mean"),
    MAE_Std=("MAE", "std"),
    RMSE_Mean=("RMSE", "mean"),
    RMSE_Std=("RMSE", "std"),
    R2_Mean=("R2", "mean"),
    R2_Std=("R2", "std")
).reset_index()

summary.to_csv(
    ARTIFACT_DIR / "segment_validation_summary.csv",
    index=False
)


# ==================================================
# 11. FINAL COMPARISON
# ==================================================

print("\n" + "=" * 90)
print("FINAL SEGMENT-WISE VALIDATION SUMMARY")
print("=" * 90)

print(
    summary.round(4).to_string(index=False)
)

print("\nSaved:")
print(ARTIFACT_DIR / "segment_validation_results.csv")
print(ARTIFACT_DIR / "segment_validation_summary.csv")

print("\nSEGMENT-WISE VALIDATION COMPLETED!")
