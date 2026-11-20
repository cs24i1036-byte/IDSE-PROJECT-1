
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "car_details_cleaned.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "car_details_deduplicated.csv"
)

df = pd.read_csv(INPUT_FILE)

print("\n===== DATASET DUPLICATE SUMMARY =====")

original_rows = len(df)

# Count rows belonging to duplicate groups
duplicate_group_rows = df.duplicated(keep=False).sum()

# Count additional rows that drop_duplicates will remove
extra_duplicate_rows = df.duplicated().sum()

# Create a separate deduplicated dataset
df_deduplicated = df.drop_duplicates().copy()

print("\nOriginal rows:", original_rows)
print("Rows in duplicate groups:", duplicate_group_rows)
print("Extra duplicate rows:", extra_duplicate_rows)
print("Rows after deduplication:", len(df_deduplicated))

print(
    "Percentage of rows removed:",
    round(extra_duplicate_rows / original_rows * 100, 2),
    "%"
)

# Investigate Volvo and Lexus
for car_name in [
    "Volvo XC40 D4 Inscription BSIV",
    "Lexus ES 300h"
]:

    car_data = df[
        df["name"].eq(car_name)
    ]

    print(f"\n===== {car_name} =====")

    print("Original records:", len(car_data))
    print("Unique complete records:", len(car_data.drop_duplicates()))
    print("Extra duplicate records:", car_data.duplicated().sum())

# Save separately; original remains untouched
df_deduplicated.to_csv(OUTPUT_FILE, index=False)

print("\nDeduplicated dataset saved at:")
print(OUTPUT_FILE)

print("\n===== COMPLETED =====")
