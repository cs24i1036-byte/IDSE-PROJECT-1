
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    RandomizedSearchCV
)

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
FIGURES_DIR = BASE_DIR / "reports" / "figures"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_FILE = ARTIFACTS_DIR / "final_model.joblib"
PREDICTIONS_FILE = ARTIFACTS_DIR / "final_predictions.csv"
IMPORTANCE_FILE = ARTIFACTS_DIR / "feature_importance.csv"
METRICS_FILE = ARTIFACTS_DIR / "final_metrics.csv"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("\n========== CARWISE MODEL EVALUATION ==========")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. DEFINE FEATURES AND TARGET
# --------------------------------------------------

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

all_features = numerical_features + categorical_features

X = df[all_features].copy()
y = df["selling_price"].copy()


# --------------------------------------------------
# 4. CREATE GROUPS
# --------------------------------------------------

# Identical feature combinations must stay together.
groups = pd.util.hash_pandas_object(
    X,
    index=False
)


# --------------------------------------------------
# 5. CREATE FINAL HOLDOUT SET
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=2026
)

development_indices, holdout_indices = next(
    splitter.split(X, y, groups=groups)
)

X_dev = X.iloc[development_indices]
X_holdout = X.iloc[holdout_indices]

y_dev = y.iloc[development_indices]
y_holdout = y.iloc[holdout_indices]

groups_dev = groups.iloc[development_indices]


print("\nDevelopment records:", len(X_dev))
print("Final holdout records:", len(X_holdout))


# --------------------------------------------------
# 6. PREPROCESSING PIPELINE
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
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
# 7. RANDOM FOREST PIPELINE
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=1
            )
        )
    ]
)


# --------------------------------------------------
# 8. HYPERPARAMETER TUNING
# --------------------------------------------------

parameter_grid = {
    "model__n_estimators": [150, 200, 300],
    "model__max_depth": [None, 15, 20, 25],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [0.7, 1.0]
}

cv = GroupKFold(n_splits=3)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_grid,
    n_iter=8,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=True
)

print("\nStarting hyperparameter tuning...")

search.fit(
    X_dev,
    y_dev,
    groups=groups_dev
)

best_model = search.best_estimator_

print("\nBest parameters:")
print(search.best_params_)

print(
    "\nBest cross-validation RMSE:",
    round(-search.best_score_, 2)
)


# --------------------------------------------------
# 9. TRAINING AND HOLDOUT PREDICTIONS
# --------------------------------------------------

train_predictions = best_model.predict(X_dev)

holdout_predictions = best_model.predict(X_holdout)


# --------------------------------------------------
# 10. EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual, predicted)

    rmse = np.sqrt(mse)

    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


train_metrics = calculate_metrics(
    y_dev,
    train_predictions
)

holdout_metrics = calculate_metrics(
    y_holdout,
    holdout_predictions
)


print("\n========== TRAINING PERFORMANCE ==========")

for metric, value in train_metrics.items():
    print(f"{metric}: {value:,.4f}")


print("\n========== FINAL HOLDOUT PERFORMANCE ==========")

for metric, value in holdout_metrics.items():
    print(f"{metric}: {value:,.4f}")


# --------------------------------------------------
# 11. SAVE METRICS
# --------------------------------------------------

metrics_df = pd.DataFrame([
    {"Dataset": "Development", **train_metrics},
    {"Dataset": "Final Holdout", **holdout_metrics}
])

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# --------------------------------------------------
# 12. ACTUAL VS PREDICTED GRAPH
# --------------------------------------------------

sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 7))

sns.scatterplot(
    x=y_holdout,
    y=holdout_predictions,
    alpha=0.5
)

minimum = min(y_holdout.min(), holdout_predictions.min())
maximum = max(y_holdout.max(), holdout_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual vs Predicted Car Prices")
plt.xlabel("Actual Selling Price (INR)")
plt.ylabel("Predicted Selling Price (INR)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "09_actual_vs_predicted.png",
    dpi=300
)

plt.show()
plt.close()


# --------------------------------------------------
# 13. RESIDUAL ANALYSIS
# --------------------------------------------------

residuals = y_holdout - holdout_predictions

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=holdout_predictions,
    y=residuals,
    alpha=0.5
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Residuals vs Predicted Prices")
plt.xlabel("Predicted Selling Price (INR)")
plt.ylabel("Residual (Actual - Predicted)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "10_residual_plot.png",
    dpi=300
)

plt.show()
plt.close()


# Residual distribution
plt.figure(figsize=(9, 6))

sns.histplot(
    residuals,
    bins=40,
    kde=True
)

plt.title("Distribution of Prediction Errors")
plt.xlabel("Residual (INR)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "11_residual_distribution.png",
    dpi=300
)

plt.show()
plt.close()


# --------------------------------------------------
# 14. FEATURE IMPORTANCE
# --------------------------------------------------

fitted_preprocessor = best_model.named_steps["preprocessor"]

fitted_model = best_model.named_steps["model"]

feature_names = fitted_preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": fitted_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)

top_features = importance_df.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(11, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "12_feature_importance.png",
    dpi=300
)

plt.show()
plt.close()


# --------------------------------------------------
# 15. SAVE FINAL PREDICTIONS
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "Actual_Price": y_holdout.values,
    "Predicted_Price": holdout_predictions,
    "Residual": residuals.values,
    "Absolute_Error": np.abs(residuals.values)
})

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# --------------------------------------------------
# 16. SAVE FINAL MODEL
# --------------------------------------------------

joblib.dump(
    best_model,
    FINAL_MODEL_FILE
)


# --------------------------------------------------
# 17. FINAL SUMMARY
# --------------------------------------------------

print("\n========== FINAL SUMMARY ==========")

print("Best model: Random Forest")

print("Cross-validation RMSE:", round(-search.best_score_, 2))

print("Final holdout RMSE:", round(holdout_metrics["RMSE"], 2))

print("Final holdout R2:", round(holdout_metrics["R2"], 4))

print("\nSaved files:")

print("Final model:", FINAL_MODEL_FILE)
print("Metrics:", METRICS_FILE)
print("Predictions:", PREDICTIONS_FILE)
print("Feature importance:", IMPORTANCE_FILE)

print("\nPhase 5 evaluation completed!")
