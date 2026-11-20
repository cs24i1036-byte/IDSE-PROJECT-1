
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR / "data" / "processed"
    / "carwise_v3_1_model_data.csv"
)

ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "car_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

CATEGORICAL_FEATURES = [
    "brand",
    "car_model",
    "variant_details",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

V1_GROUP_FEATURES = [
    "car_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

TARGET = "selling_price"

ALL_FEATURES = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)


# ============================================================
# 3. PREPROCESSING
# ============================================================

def build_preprocessor(dense_output=False):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
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
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=not dense_output
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ],
        sparse_threshold=0.0 if dense_output else 0.3
    )

    return preprocessor


# ============================================================
# 4. ALGORITHM DEFINITIONS
# ============================================================

def get_model_configurations():

    configurations = {

        "Random Forest": {
            "model": RandomForestRegressor(
                random_state=42,
                n_jobs=1
            ),

            "dense": False,

            "params": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [None, 15, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": [
                    "sqrt", 0.7, 1.0
                ]
            }
        },

        "Extra Trees": {
            "model": ExtraTreesRegressor(
                random_state=42,
                n_jobs=1
            ),

            "dense": False,

            "params": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [None, 15, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": [
                    "sqrt", 0.7, 1.0
                ]
            }
        },

        "Hist Gradient Boosting": {
            "model": HistGradientBoostingRegressor(
                random_state=42,
                early_stopping=True
            ),

            "dense": True,

            "params": {
                "model__max_iter": [200, 300, 400],
                "model__learning_rate": [
                    0.03, 0.05, 0.08, 0.1
                ],
                "model__max_leaf_nodes": [
                    15, 31, 63
                ],
                "model__min_samples_leaf": [
                    10, 20, 30
                ],
                "model__l2_regularization": [
                    0.0, 0.1, 1.0
                ]
            }
        }
    }

    return configurations


# ============================================================
# 5. BUILD PIPELINE
# ============================================================

def build_pipeline(model, dense_output=False):

    preprocessor = build_preprocessor(
        dense_output=dense_output
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


# ============================================================
# 6. EVALUATION METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("CARWISE V3.1 ALGORITHM COMPARISON")
    print("=" * 65)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(DATA_FILE)

    print("\nDataset shape:", df.shape)

    X = df[ALL_FEATURES].copy()

    y = df[TARGET].copy()

    # --------------------------------------------------------
    # CREATE GROUPS
    # --------------------------------------------------------

    # Keep identical V1 feature combinations together.
    groups = pd.util.hash_pandas_object(
        X[V1_GROUP_FEATURES].astype(str),
        index=False
    ).values

    print(
        "Unique groups:",
        len(np.unique(groups))
    )

    # --------------------------------------------------------
    # RESERVE UNTOUCHED TEST SET
    # --------------------------------------------------------

    holdout_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=2027
    )

    train_idx, test_idx = next(
        holdout_splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    print("\nDevelopment rows:", len(X_train))
    print("Untouched test rows:", len(X_test))

    print(
        "Development groups:",
        len(np.unique(groups_train))
    )

    print(
        "Test groups:",
        len(np.unique(groups_test))
    )

    # Verify no group overlap.
    overlap = set(groups_train).intersection(
        set(groups_test)
    )

    if overlap:
        raise ValueError(
            "Group leakage detected between development and test data."
        )

    print("\nGroup separation verified.")

    # --------------------------------------------------------
    # GROUPED CROSS-VALIDATION
    # --------------------------------------------------------

    cv = GroupKFold(n_splits=5)

    configurations = get_model_configurations()

    cv_results = []
    test_results = []
    test_predictions = []

    best_overall_model = None
    best_overall_name = None
    best_overall_mae = float("inf")

    # --------------------------------------------------------
    # TRAIN AND TUNE EACH ALGORITHM
    # --------------------------------------------------------

    for model_name, config in configurations.items():

        print("\n" + "=" * 65)
        print("TRAINING:", model_name)
        print("=" * 65)

        pipeline = build_pipeline(
            model=config["model"],
            dense_output=config["dense"]
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["params"],
            n_iter=12,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            refit=True,
            return_train_score=False,
            error_score="raise"
        )

        # Groups are used only inside development data.
        search.fit(
            X_train,
            y_train,
            groups=groups_train
        )

        # ----------------------------------------------------
        # CROSS-VALIDATION RESULTS
        # ----------------------------------------------------

        best_cv_mae = -search.best_score_

        print("\nBest CV MAE: ₹{:,.2f}".format(
            best_cv_mae
        ))

        print("\nBest parameters:")

        print(search.best_params_)

        cv_results.append({
            "Algorithm": model_name,
            "Best_CV_MAE": best_cv_mae,
            "Best_Parameters": str(search.best_params_)
        })

        # ----------------------------------------------------
        # UNTOUCHED TEST EVALUATION
        # ----------------------------------------------------

        best_model = search.best_estimator_

        predictions = best_model.predict(X_test)

        metrics = calculate_metrics(
            y_test,
            predictions
        )

        print("\nUntouched Test Results:")

        print("MAE:  ₹{:,.2f}".format(metrics["MAE"]))
        print("RMSE: ₹{:,.2f}".format(metrics["RMSE"]))
        print("R2:   {:.4f}".format(metrics["R2"]))

        test_results.append({
            "Algorithm": model_name,
            **metrics
        })

        prediction_df = pd.DataFrame({
            "Algorithm": model_name,
            "Actual_Price": y_test.values,
            "Predicted_Price": predictions,
            "Residual": (
                y_test.values - predictions
            ),
            "Absolute_Error": np.abs(
                y_test.values - predictions
            )
        })

        test_predictions.append(prediction_df)

        # ----------------------------------------------------
        # SAVE EACH SEARCH RESULT
        # ----------------------------------------------------

        search_results_df = pd.DataFrame(
            search.cv_results_
        )

        safe_name = (
            model_name.lower()
            .replace(" ", "_")
        )

        search_results_df.to_csv(
            ARTIFACT_DIR
            / f"{safe_name}_search_results.csv",
            index=False
        )

        # ----------------------------------------------------
        # TRACK WINNER BY CV MAE ONLY
        # ----------------------------------------------------

        if best_cv_mae < best_overall_mae:

            best_overall_mae = best_cv_mae

            best_overall_model = best_model

            best_overall_name = model_name

    # ========================================================
    # 8. SAVE COMPARISON RESULTS
    # ========================================================

    cv_df = pd.DataFrame(cv_results)

    test_df = pd.DataFrame(test_results)

    predictions_df = pd.concat(
        test_predictions,
        ignore_index=True
    )

    cv_df.to_csv(
        ARTIFACT_DIR / "v3_1_algorithm_cv_results.csv",
        index=False
    )

    test_df.to_csv(
        ARTIFACT_DIR / "v3_1_algorithm_test_results.csv",
        index=False
    )

    predictions_df.to_csv(
        ARTIFACT_DIR / "v3_1_algorithm_predictions.csv",
        index=False
    )

    # ========================================================
    # 9. FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 65)
    print("CROSS-VALIDATION COMPARISON")
    print("=" * 65)

    print(
        cv_df[
            ["Algorithm", "Best_CV_MAE"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("UNTOUCHED TEST COMPARISON")
    print("=" * 65)

    print(
        test_df.sort_values("MAE").to_string(
            index=False
        )
    )

    # ========================================================
    # 10. SAVE SELECTED DEVELOPMENT MODEL
    # ========================================================

    winner_path = (
        ARTIFACT_DIR
        / "v3_1_best_algorithm_dev_model.joblib"
    )

    joblib.dump(
        best_overall_model,
        winner_path
    )

    print("\n" + "=" * 65)
    print("ALGORITHM SELECTION COMPLETE")
    print("=" * 65)

    print("Selected algorithm:", best_overall_name)

    print(
        "Best grouped CV MAE: ₹{:,.2f}".format(
            best_overall_mae
        )
    )

    print("\nSelected development model saved:")
    print(winner_path)

    print("\nSaved result files:")
    print("1. v3_1_algorithm_cv_results.csv")
    print("2. v3_1_algorithm_test_results.csv")
    print("3. v3_1_algorithm_predictions.csv")
    print("4. Individual algorithm search results")

    print("\nIMPORTANT:")
    print(
        "The selected model is fitted on development data only."
    )

    print(
        "Do not use the test set to tune or select hyperparameters."
    )

    print("\n===== EXPERIMENT COMPLETED =====")


if __name__ == "__main__":
    main()
