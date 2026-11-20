
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FILE = BASE_DIR / "data" / "processed" / "car_details_cleaned.csv"

df = pd.read_csv(FILE)

print("\n===== DUPLICATE INVESTIGATION =====\n")

# Check exact duplicate rows across all columns
exact_duplicates = df.duplicated(keep=False)

print("Total rows:", len(df))
print("Rows belonging to exact duplicate groups:", exact_duplicates.sum())

# Inspect Volvo records
volvo = df[
    df["name"].str.contains("Volvo XC40 D4 Inscription BSIV", case=False, na=False)
].copy()

print("\n===== VOLVO XC40 RECORDS =====\n")

print("Number of records:", len(volvo))

print(
    volvo[
        ["name", "year", "selling_price", "km_driven",
         "fuel", "transmission", "owner"]
    ].to_string(index=False)
)

# Check duplicates of the Volvo records
print("\nExact duplicate Volvo rows:")
print(volvo.duplicated(keep=False).sum())

print("\nUnique Volvo configurations:")
print(volvo.drop_duplicates().shape[0])

# Check Lexus records
lexus = df[
    df["name"].str.contains("Lexus ES 300h", case=False, na=False)
].copy()

print("\n===== LEXUS ES 300h RECORDS =====\n")

print("Number of records:", len(lexus))

print(
    lexus[
        ["name", "year", "selling_price", "km_driven",
         "fuel", "transmission", "owner"]
    ].head(15).to_string(index=False)
)

print("\nExact duplicate Lexus rows:")
print(lexus.duplicated(keep=False).sum())

print("\n===== INVESTIGATION COMPLETED =====")
