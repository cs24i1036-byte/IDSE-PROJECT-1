
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent

FILE = BASE_DIR / "artifacts" / "v1_vs_v2_predictions.csv"

if not FILE.exists():
    print("ERROR: V1 vs V2 predictions file not found!")
    raise SystemExit(1)

df = pd.read_csv(FILE)

# Price segments based on actual selling price
bins = [0, 500000, 1500000, 3000000, np.inf]

labels = [
    "Budget",
    "Mid-range",
    "Premium",
    "Luxury"
]

df["Price_Segment"] = pd.cut(
    df["Actual_Price"],
    bins=bins,
    labels=labels,
    right=False
)

results = []

for segment, group in df.groupby("Price_Segment", observed=False):

    if len(group) == 0:
        continue

    actual = group["Actual_Price"]

    for model_name, prediction_column in [
        ("V1", "V1_Predicted_Price"),
        ("V2", "V2_Predicted_Price")
    ]:

        predicted = group[prediction_column]

        mae = mean_absolute_error(actual, predicted)

        rmse = np.sqrt(
            mean_squared_error(actual, predicted)
        )

        r2 = (
            r2_score(actual, predicted)
            if len(group) >= 2 and actual.nunique() > 1
            else np.nan
        )

        results.append({
            "Segment": str(segment),
            "Model": model_name,
            "Records": len(group),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

results_df = pd.DataFrame(results)

print("\n===== V1 vs V2: SEGMENT-WISE COMPARISON =====\n")

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": lambda x: f"₹{x:,.2f}",
            "RMSE": lambda x: f"₹{x:,.2f}",
            "R2": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        }
    )
)

# Calculate percentage improvement in MAE and RMSE
print("\n===== V2 IMPROVEMENT BY SEGMENT =====\n")

for segment in labels:

    subset = results_df[
        results_df["Segment"] == segment
    ]

    if len(subset) != 2:
        continue

    v1 = subset[subset["Model"] == "V1"].iloc[0]
    v2 = subset[subset["Model"] == "V2"].iloc[0]

    mae_improvement = (
        (v1["MAE"] - v2["MAE"]) / v1["MAE"] * 100
    )

    rmse_improvement = (
        (v1["RMSE"] - v2["RMSE"]) / v1["RMSE"] * 100
    )

    print(f"{segment}:")
    print(f"  MAE improvement: {mae_improvement:.2f}%")
    print(f"  RMSE improvement: {rmse_improvement:.2f}%")

# Save report
output = BASE_DIR / "artifacts" / "v1_vs_v2_segment_comparison.csv"

results_df.to_csv(output, index=False)

print("\nSaved:", output)
print("\n===== COMPARISON COMPLETED =====")
