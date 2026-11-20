
import pandas as pd
import numpy as np
import re
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "car_details_deduplicated.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "carwise_v3_1_model_data.csv"
)

AUDIT_DIR = BASE_DIR / "artifacts"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MODEL DICTIONARY
# Each brand has its own known model families.
# Longest-prefix matching will be used.
# ============================================================

MODEL_DICTIONARY = {

    "Maruti": [
        "Swift Dzire", "Swift", "Wagon R",
        "Alto 800", "Alto K10", "Alto",
        "800", "Omni", "Ertiga", "Baleno",
        "Ritz", "Celerio", "Zen", "SX4",
        "Eeco", "Esteem", "Gypsy","XL6",
        "Vitara Brezza", "S-Cross", "Ciaz",
        "Ignis", "Kizashi", "Grand Vitara",
        "A-Star", "Versa", "Reno", "Dzire",
        "S-Presso", "Estilo", "Zen Estilo",
        "Swift Dzire Tour"
    ],

    "Hyundai": [
        "Grand i10", "Elite i20", "i20",
        "i10", "Santro Xing", "Santro",
        "Verna", "Creta", "EON", "Xcent",
        "Accent", "Elantra", "Getz",
        "Santa Fe", "Tucson", "Sonata",
        "Fluidic Verna", "Venue", "Terracan",
        "Alcazar", "Kona"
    ],

    "Mahindra": [
        "KUV 100", "XUV500", "XUV300",
        "Scorpio", "Bolero", "Thar",
        "Xylo", "Quanto", "TUV 300",
        "NuvoSport", "Verito", "Logan",
        "Marshal", "Ingenio", "Alturas G4",
        "Supro", "Jeep", "Classic",
        "Voyager", "Maxximo", "E Verito",
        "Marazzo", "Rexton", "Ssangyong Rexton",
        "Willys CJ", "Imperio", "Armada",
        "Commander", "Legend"
    ],

    "Tata": [
        "Indica Vista", "Indica V2", "Indica",
        "Indigo Manza", "Indigo CS", "Indigo",
        "Safari Storme", "Safari", "Nexon",
        "Tiago", "Tigor", "Harrier", "Hexa",
        "Zest", "Bolt", "Sumo Gold", "Sumo",
        "Nano", "Aria", "Venture", "Movus",
        "Winger", "Xenon", "Estate", "Spacio",
        "Magic", "Telcoline", "Manza",
        "New Safari", "Altroz", "Punch",
        "Sierra", "Curvv"
    ],

    "Toyota": [
        "Land Cruiser Prado", "Land Cruiser",
        "Innova Crysta", "Innova", "Fortuner",
        "Corolla Altis", "Corolla",
        "Etios Liva", "Etios Cross",
        "Platinum Etios", "Etios",
        "Premio", "Camry", "Qualis",
        "Yaris", "Glanza", "Prius", "Hilux",
        "Urban Cruiser", "Rumion"
    ],

    "Honda": [
        "City", "Amaze", "Jazz", "Brio",
        "Civic", "Accord", "CR-V", "WR-V",
        "BR-V", "BRV", "Mobilio",
        "Prelude", "Elevate"
    ],

    "Ford": [
        "EcoSport", "Endeavour",
        "Figo Aspire", "Figo", "Aspire",
        "Fiesta Classic", "Fiesta",
        "Ikon", "Freestyle", "Fusion",
        "Mondeo", "Classic", "Mustang"
    ],

    "Chevrolet": [
        "Trailblazer", "Enjoy", "Beat",
        "Spark", "Cruze", "Tavera",
        "Sail", "Aveo", "Optra",
        "Captiva", "Forester"
    ],

    "Renault": [
        "Duster", "KWID", "Kwid",
        "Lodgy", "Scala", "Pulse","Captur",
        "Koleos", "Fluence", "Triber",
        "Kiger"
    ],

    "Volkswagen": [
        "Polo", "Vento", "Jetta",
        "Passat", "Ameo", "Tiguan","CrossPolo",
        "Touareg", "Multivan", "GTI",
        "Beetle"
    ],

    "Skoda": [
        "Rapid", "Superb", "Octavia",
        "Laura", "Fabia", "Yeti",
        "Kodiaq", "Karoq"
    ],

    "BMW": [
        "3 Series", "5 Series", "6 Series",
        "7 Series", "X1", "X2", "X3",
        "X4", "X5", "X6", "X7", "Z4"
    ],

    "Mercedes-Benz": [
        "E-Class", "C-Class", "S-Class",
        "B-Class", "B Class", "GL-Class",
        "GLA", "GLC", "GLE", "GLS",
        "M-Class", "A-Class", "B-Class",
        "CLA", "CLS", "SLK", "G-Class",
        "New C-Class", "New E-Class",
        "New S-Class", "R-Class"
    ],

    "Audi": [
        "A3", "A4", "A6", "A7",
        "A8", "Q3", "Q5", "Q7",
        "Q8", "TT", "RS5"
    ],

    "Nissan": [
        "Micra", "Sunny", "Terrano",
        "Evalia", "Teana", "Kicks",
        "370Z"
    ],

    "Land Rover": [
        "Range Rover Evoque",
        "Range Rover Sport",
        "Range Rover",
        "Discovery Sport",
        "Discovery",
        "Freelander"
    ],

    "Volvo": [
        "XC40", "XC60", "XC90",
        "S60", "S80", "S90",
        "V40", "V90"
    ],

    "Jeep": [
        "Compass", "Wrangler",
        "Grand Cherokee"
    ],

    "Datsun": [
        "RediGO", "Redi Go",
        "GO+", "GO"
    ],

    "Fiat": [
        "Palio", "Punto", "Linea",
        "Grande Punto", "Avventura",
        "Abarth", "Petra"
    ],

    "Isuzu": [
        "MU-7", "MU 7", "MUX",
        "MU-X", "D-Max", "MU"
    ],

    "Peugeot": [
        "309", "508"
    ],

    "Ashok": [
        "Leyland Stile", "Stile"
    ],

    "Mitsubishi": [
        "Pajero Sport", "Pajero",
        "Outlander", "Lancer",
        "Cedia", "Montero"
    ],

    "Jaguar": [
        "XF", "XE", "XJ",
        "F-Pace", "F-Type",
        "XK"
    ],

    "MG": [
        "Hector", "ZS EV",
        "Gloster", "Astor",
        "Comet", "Hector Plus"
    ],

    "Kia": [
        "Seltos", "Sonet",
        "Carnival", "Carens",
        "EV6"
    ],

    "Lexus": [
        "ES", "RX", "NX",
        "LS", "LX"
    ],

    "Force": [
        "One", "Gurkha",
        "Trax", "Traveller"
    ],

    "Ambassador": [
        "Classic", "Grand",
        "Avigo", "Mark"
    ],

    "Daewoo": [
        "Matiz", "Cielo",
        "Nexia"
    ],

    "Opel": [
        "Corsa", "Astra",
        "Vectra"
    ]
}


# ============================================================
# 3. BRAND-SPECIFIC PREFIX ALIASES
# These remove descriptive words before model matching.
# ============================================================

PREFIX_ALIASES = {

    "Hyundai": [
        "Elite "
    ],

    "Tata": [
        "New "
    ],

    "Mercedes-Benz": [
        "New "
    ],

    "Mahindra": [
        "Ssangyong "
    ]
}


# ============================================================
# 4. TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    value = re.sub(r"\s+", " ", value)

    return value


# ============================================================
# 5. MODEL AND VARIANT EXTRACTION
# ============================================================

def extract_model_variant(row):

    brand = normalize_text(row["brand"])

    name = normalize_text(row["name"])

    remaining_name = name

    # Remove brand prefix.
    if remaining_name.lower().startswith(brand.lower()):

        remaining_name = remaining_name[len(brand):].strip()

    # Special case: Mahindra Renault Logan.
    if (
        brand == "Mahindra"
        and remaining_name.lower().startswith("renault logan")
    ):

        remaining_name = remaining_name[len("Renault "):].strip()

    # Remove known descriptive prefixes.
    aliases = PREFIX_ALIASES.get(brand, [])

    for prefix in aliases:

        if remaining_name.lower().startswith(prefix.lower()):

            remaining_name = remaining_name[len(prefix):].strip()

            break

    # Get models for this brand.
    possible_models = MODEL_DICTIONARY.get(brand, [])

    # Longest model name first.
    possible_models = sorted(
        possible_models,
        key=len,
        reverse=True
    )

    remaining_lower = remaining_name.lower()

    for model in possible_models:

        model_lower = model.lower()

        if remaining_lower == model_lower:

            return pd.Series([model, "Unspecified"])

        if remaining_lower.startswith(model_lower):

            next_index = len(model)

            # Require a word boundary.
            if (
                len(remaining_name) == next_index
                or remaining_name[next_index].isspace()
            ):

                variant = remaining_name[next_index:].strip()

                if not variant:
                    variant = "Unspecified"

                return pd.Series([model, variant])

    # Fallback: preserve unmatched name.
    if not remaining_name:

        remaining_name = "Unknown"

    return pd.Series([
        remaining_name,
        "Unspecified"
    ])


# ============================================================
# 6. MAIN FEATURE ENGINEERING
# ============================================================

def main():

    print("\n===== CARWISE V3.1 FEATURE ENGINEERING =====")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print("Original dataset shape:", df.shape)

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    numeric_columns = [
        "mileage",
        "engine",
        "max_power",
        "seats"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Invalid measurements become missing values.
    for col in ["mileage", "engine", "max_power"]:

        df.loc[df[col] <= 0, col] = np.nan

    # --------------------------------------------------------
    # CAR AGE
    # --------------------------------------------------------

    # Historical dataset snapshot assumption.
    df["car_age"] = 2020 - df["year"]

    # --------------------------------------------------------
    # MODEL AND VARIANT EXTRACTION
    # --------------------------------------------------------

    df[["car_model", "variant_details"]] = df.apply(
        extract_model_variant,
        axis=1
    )

    df["car_model"] = df["car_model"].apply(normalize_text)

    df["variant_details"] = (
        df["variant_details"].apply(normalize_text)
    )

    # --------------------------------------------------------
    # SELECT FEATURES
    # --------------------------------------------------------

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
        "car_model",
        "variant_details",
        "fuel",
        "seller_type",
        "transmission",
        "owner"
    ]

    target = "selling_price"

    selected_columns = (
        numerical_features
        + categorical_features
        + [target]
    )

    model_df = df[selected_columns].copy()

    # --------------------------------------------------------
    # SAVE PROCESSED DATASET
    # --------------------------------------------------------

    model_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # 7. DATASET SUMMARY
    # ========================================================

    print("\n===== DATASET SUMMARY =====")

    print("Final dataset shape:", model_df.shape)

    print(
        "Unique car models:",
        model_df["car_model"].nunique()
    )

    print(
        "Unique variants:",
        model_df["variant_details"].nunique()
    )

    print("\nMissing values:")

    print(model_df.isnull().sum())

    # ========================================================
    # 8. UNSPECIFIED VARIANT AUDIT
    # ========================================================

    print("\n===== UNSPECIFIED VARIANT AUDIT =====")

    unspecified_df = df[
        df["variant_details"] == "Unspecified"
    ][["name", "brand", "car_model"]].copy()

    unspecified_count = len(unspecified_df)

    unspecified_percentage = (
        unspecified_count / len(df)
    ) * 100

    print("Unspecified records:", unspecified_count)

    print(
        "Unspecified percentage:",
        round(unspecified_percentage, 2),
        "%"
    )

    if unspecified_count > 0:

        print("\nUnspecified records by brand:")

        print(unspecified_df["brand"].value_counts())

        print("\nUnspecified examples:")

        print(
            unspecified_df.head(50).to_string(index=False)
        )

    unspecified_df.to_csv(
        AUDIT_DIR / "unspecified_variant_audit.csv",
        index=False
    )

    # ========================================================
    # 9. MODEL FAMILY CHECK
    # ========================================================

    print("\n===== MODEL FAMILY CHECK =====")

    model_checks = [
        ("Maruti", "Alto 800"),
        ("Maruti", "Alto"),
        ("Maruti", "Alto K10"),
        ("Maruti", "Swift Dzire"),
        ("Maruti", "Swift"),
        ("Maruti", "Wagon R"),
        ("Maruti", "Omni"),
        ("Hyundai", "Grand i10"),
        ("Hyundai", "i20"),
        ("BMW", "3 Series"),
        ("BMW", "5 Series"),
        ("BMW", "X4"),
        ("Land Rover", "Range Rover Evoque"),
        ("Toyota", "Land Cruiser Prado"),
        ("Toyota", "Platinum Etios"),
        ("Tata", "Tiago"),
        ("Tata", "Safari"),
        ("Ford", "Aspire"),
        ("Honda", "BRV"),
        ("Datsun", "GO"),
        ("Datsun", "RediGO"),
        ("Fiat", "Palio"),
        ("Audi", "A6"),
        ("Jeep", "Compass"),
        ("Volvo", "XC40"),
        ("Volvo", "S90"),
        ("Mahindra", "Logan"),
        ("Mahindra", "Marazzo"),
        ("Mitsubishi", "Pajero Sport"),
        ("Jaguar", "XF"),
        ("MG", "Hector"),
        ("Kia", "Seltos"),
        ("Lexus", "ES")
    ]

    for brand, model in model_checks:

        count = len(
            df[
                (df["brand"] == brand)
                & (df["car_model"] == model)
            ]
        )

        print(f"{brand} {model}: {count} records")

    # ========================================================
    # 10. TOP 30 MODELS
    # ========================================================

    print("\n===== TOP 30 MODELS =====")

    print(
        df["car_model"]
        .value_counts()
        .head(30)
    )

    # ========================================================
    # 11. SAMPLE EXTRACTIONS
    # ========================================================

    print("\n===== SAMPLE EXTRACTIONS =====")

    sample_names = [
        "Maruti Swift Dzire VDI",
        "Skoda Rapid 1.5 TDI Ambition",
        "Honda City 2017-2020 EXi",
        "Hyundai Elite i20 Sportz Plus BSIV",
        "Maruti Wagon R LXI DUO BSIII",
        "Maruti Alto 800 LXI",
        "BMW X4 M Sport X xDrive20d",
        "Mercedes-Benz B Class B180",
        "Mercedes-Benz New C-Class 220 CDI AT",
        "Mitsubishi Pajero Sport 4X4",
        "Jaguar XF 2.0 Diesel Portfolio",
        "MG Hector Sharp AT BSIV",
        "Kia Seltos HTE D",
        "Lexus ES 300h",
        "Tata New Safari DICOR 2.2 EX 4x2",
        "Datsun RediGO 1.0 S",
        "Maruti Estilo LXI",
        "Mahindra Renault Logan 1.4 GLX Petrol",
        "Mahindra Marazzo M6 8Str"
    ]

    sample_df = pd.DataFrame({
        "name": sample_names
    })

    known_brands = sorted(
        df["brand"].dropna().unique(),
        key=len,
        reverse=True
    )

    def get_sample_brand(name):

        for brand in known_brands:

            if name.lower().startswith(brand.lower()):

                return brand

        return "Unknown"

    sample_df["brand"] = sample_df["name"].apply(
        get_sample_brand
    )

    sample_df[["car_model", "variant_details"]] = (
        sample_df.apply(
            extract_model_variant,
            axis=1
        )
    )

    print(
        sample_df[
            [
                "name",
                "brand",
                "car_model",
                "variant_details"
            ]
        ].to_string(index=False)
    )

    sample_df.to_csv(
        AUDIT_DIR / "v3_1_sample_extractions.csv",
        index=False
    )

    # ========================================================
    # 12. SAVE MODEL FREQUENCY AUDIT
    # ========================================================

    model_frequency = (
        df["car_model"]
        .value_counts()
        .rename_axis("car_model")
        .reset_index(name="count")
    )

    model_frequency.to_csv(
        AUDIT_DIR / "v3_1_model_frequency.csv",
        index=False
    )

    # ========================================================
    # 13. FINAL STATUS
    # ========================================================

    print("\n===== FEATURE ENGINEERING COMPLETE =====")

    print("Saved dataset:", OUTPUT_FILE)

    print(
        "Saved unspecified audit:",
        AUDIT_DIR / "unspecified_variant_audit.csv"
    )

    print(
        "Saved sample audit:",
        AUDIT_DIR / "v3_1_sample_extractions.csv"
    )


if __name__ == "__main__":
    main()
