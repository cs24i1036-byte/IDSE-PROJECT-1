
import pandas as pd
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Final predictions file
FILE = BASE_DIR / "artifacts" / "final_predictions.csv"

# Check file existence
if not FILE.exists():
    print("ERROR: final_predictions.csv not found!")
    raise SystemExit(1)

# Load predictions
df = pd.read_csv(FILE)

print("\n===== CARWISE INDIA: ERROR ANALYSIS =====\n")

print("Available columns:")
print(df.columns.tolist())

# Correct column names from your CSV
actual_col = "Actual_Price"
predicted_col = "Predicted_Price"

# Calculate prediction errors
df["Error"] = df[actual_col] - df[predicted_col]
df["Absolute_Error"] = df["Error"].abs()

# Error statistics
print("\n===== ERROR STATISTICS =====")
print(df["Absolute_Error"].describe())

# 10 largest prediction errors
print("\n===== 10 LARGEST PREDICTION ERRORS =====")

print(
    df.nlargest(10, "Absolute_Error")[
        [actual_col, predicted_col, "Error", "Absolute_Error"]
    ].to_string(index=False)
)

# Actual vs predicted average
print("\n===== PRICE COMPARISON =====")

print("Mean actual price:",
      round(df[actual_col].mean(), 2))

print("Mean predicted price:",
      round(df[predicted_col].mean(), 2))

print("\n===== ERROR ANALYSIS COMPLETED =====")
