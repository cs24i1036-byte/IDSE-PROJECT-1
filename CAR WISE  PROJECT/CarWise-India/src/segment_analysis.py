
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input and output paths
INPUT_FILE = BASE_DIR / "artifacts" / "final_predictions.csv"
OUTPUT_FILE = BASE_DIR / "artifacts" / "segment_analysis.csv"

# Check input file
if not INPUT_FILE.exists():
    print("ERROR: final_predictions.csv not found!")
    raise SystemExit(1)

# Load final holdout predictions
df = pd.read_csv(INPUT_FILE)

# Correct column names
actual_col = "Actual_Price"
predicted_col = "Predicted_Price"

if actual_col not in df.columns or predicted_col not in df.columns:
    print("ERROR: Required columns not found.")
    print("Available columns:", df.columns.tolist())
    raise SystemExit(1)

# Remove invalid records, if any
df = df[[actual_col, predicted_col]].dropna()

df = df[
    (df[actual_col] > 0) &
    (df[predicted_col].notna())
].copy()

# Calculate errors
df["Residual"] = df[actual_col] - df[predicted_col]
df["Absolute_Error"] = df["Residual"].abs()

# Divide cars into price segments based on ACTUAL price
bins = [0, 500000, 1500000, 3000000, np.inf]

labels = [
    "Budget (Below 5 Lakh)",
    "Mid-range (5-15 Lakh)",
    "Premium (15-30 Lakh)",
    "Luxury (30 Lakh and Above)"
]

df["Price_Segment"] = pd.cut(
    df[actual_col],
    bins=bins,
    labels=labels,
    right=False
)

# Calculate performance for each segment
results = []

for segment, group in df.groupby("Price_Segment", observed=False):

    if len(group) == 0:
        continue

    actual = group[actual_col]
    predicted = group[predicted_col]

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    median_error = group["Absolute_Error"].median()

    mean_residual = group["Residual"].mean()

    # R2 needs at least two records and variation in actual prices
    if len(group) >= 2 and actual.nunique() > 1:
        r2 = r2_score(actual, predicted)
    else:
        r2 = np.nan

    results.append({
        "Price_Segment": str(segment),
        "Number_of_Cars": len(group),
        "MAE": mae,
        "RMSE": rmse,
        "Median_Absolute_Error": median_error,
        "Mean_Residual": mean_residual,
        "R2": r2
    })

# Create results dataframe
results_df = pd.DataFrame(results)

# Display results
print("\n" + "=" * 65)
print("          CARWISE INDIA - PRICE SEGMENT ANALYSIS")
print("=" * 65)

print("\nPerformance by Actual Selling Price Segment:\n")

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": lambda x: f"₹{x:,.2f}",
            "RMSE": lambda x: f"₹{x:,.2f}",
            "Median_Absolute_Error": lambda x: f"₹{x:,.2f}",
            "Mean_Residual": lambda x: f"₹{x:,.2f}",
            "R2": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        }
    )
)

# Save report
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("Analysis saved successfully!")
print(f"File location: {OUTPUT_FILE}")
print("=" * 65)
