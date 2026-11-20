
import pandas as pd
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Final evaluation results file
METRICS_FILE = BASE_DIR / "artifacts" / "final_metrics.csv"

# Check whether the file exists
if not METRICS_FILE.exists():
    print("ERROR: final_metrics.csv not found!")
    print("Please run Phase 5 evaluation first.")
    raise SystemExit(1)

# Load final metrics
metrics = pd.read_csv(METRICS_FILE)

print("\n" + "=" * 55)
print("          CARWISE INDIA - FINAL MODEL RESULTS")
print("=" * 55)

# Display results
print("\nFINAL MODEL PERFORMANCE:\n")
print(metrics.to_string(index=False))

print("\n" + "=" * 55)
print("                 PERFORMANCE SUMMARY")
print("=" * 55)

# Display metric information
for column in metrics.columns:
    print(f"\n{column}:")
    print(metrics[column].to_string(index=False))

print("\n" + "=" * 55)
print("              VALIDATION CHECK COMPLETED")
print("=" * 55)
