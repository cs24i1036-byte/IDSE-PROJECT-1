
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. PATHS AND DATA LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "car_details_cleaned.csv"

FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

print("\n========== CARWISE EXPLORATORY DATA ANALYSIS ==========")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. BASIC DATA UNDERSTANDING
# --------------------------------------------------

print("\n1. Dataset information:")
print(df.info())

print("\n2. Statistical summary:")
print(df.describe())

print("\n3. Missing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 3. SET VISUALIZATION STYLE
# --------------------------------------------------

sns.set_theme(style="whitegrid")

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# --------------------------------------------------
# GRAPH 1: SELLING PRICE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="selling_price",
    bins=50,
    kde=True
)

plt.title("Distribution of Used Car Selling Prices")
plt.xlabel("Selling Price (INR)")
plt.ylabel("Number of Cars")

save_plot("01_price_distribution.png")


# --------------------------------------------------
# GRAPH 2: SELLING PRICE BOXPLOT
# --------------------------------------------------

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x="selling_price"
)

plt.title("Boxplot of Used Car Selling Prices")
plt.xlabel("Selling Price (INR)")

save_plot("02_price_boxplot.png")


# --------------------------------------------------
# GRAPH 3: BRAND-WISE AVERAGE SELLING PRICE
# --------------------------------------------------

# Select the 10 brands with the most records.
top_brands = df["brand"].value_counts().head(10).index

brand_data = df[df["brand"].isin(top_brands)]

brand_prices = (
    brand_data.groupby("brand")["selling_price"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=brand_prices,
    x="selling_price",
    y="brand"
)

plt.title("Average Selling Price of Top 10 Brands by Record Count")
plt.xlabel("Average Selling Price (INR)")
plt.ylabel("Brand")

save_plot("03_brand_average_price.png")


# --------------------------------------------------
# GRAPH 4: KILOMETRES DRIVEN VS SELLING PRICE
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="km_driven",
    y="selling_price",
    alpha=0.5
)

plt.title("Kilometres Driven vs Selling Price")
plt.xlabel("Kilometres Driven")
plt.ylabel("Selling Price (INR)")

save_plot("04_km_vs_price.png")


# --------------------------------------------------
# GRAPH 5: FUEL TYPE VS SELLING PRICE
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="fuel",
    y="selling_price"
)

plt.title("Selling Price Distribution by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price (INR)")

save_plot("05_fuel_vs_price.png")


# --------------------------------------------------
# GRAPH 6: TRANSMISSION VS SELLING PRICE
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="transmission",
    y="selling_price"
)

plt.title("Selling Price Distribution by Transmission")
plt.xlabel("Transmission Type")
plt.ylabel("Selling Price (INR)")

save_plot("06_transmission_vs_price.png")


# --------------------------------------------------
# GRAPH 7: YEAR VS SELLING PRICE
# --------------------------------------------------

plt.figure(figsize=(12, 6))

sns.scatterplot(
    data=df,
    x="year",
    y="selling_price",
    alpha=0.5
)

plt.title("Manufacturing Year vs Selling Price")
plt.xlabel("Manufacturing Year")
plt.ylabel("Selling Price (INR)")

save_plot("07_year_vs_price.png")


# --------------------------------------------------
# GRAPH 8: CORRELATION HEATMAP
# --------------------------------------------------

numeric_columns = [
    "year",
    "selling_price",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")

save_plot("08_correlation_heatmap.png")


# --------------------------------------------------
# 4. ADDITIONAL NUMERICAL INSIGHTS
# --------------------------------------------------

print("\n========== EDA INSIGHTS ==========")

print("\n1. Average selling price by fuel:")
print(
    df.groupby("fuel")["selling_price"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)

print("\n2. Average selling price by transmission:")
print(
    df.groupby("transmission")["selling_price"]
    .agg(["count", "mean", "median"])
)

print("\n3. Average selling price by owner:")
print(
    df.groupby("owner")["selling_price"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)

print("\n4. Average selling price by manufacturing year:")
print(
    df.groupby("year")["selling_price"]
    .agg(["count", "mean", "median"])
    .sort_index()
)

print("\n5. Correlation with selling price:")
print(
    correlation["selling_price"]
    .sort_values(ascending=False)
)

print("\nAll EDA graphs saved successfully!")
print("Location:", FIGURES_DIR)
