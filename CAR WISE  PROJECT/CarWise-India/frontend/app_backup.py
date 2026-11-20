
import streamlit as st
import pandas as pd
import requests
import joblib

from pathlib import Path


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CarWise India | Smart Car Valuation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed" /
    "carwise_v3_1_model_data.csv"
)

MODEL_PATH = (
    BASE_DIR / "artifacts" /
    "v3_1_random_forest_final.joblib"
)

TEST_RESULTS_PATH = (
    BASE_DIR / "artifacts" /
    "v3_1_algorithm_test_results.csv"
)

CV_RESULTS_PATH = (
    BASE_DIR / "artifacts" /
    "v3_1_algorithm_cv_results.csv"
)

API_URL = "http://127.0.0.1:8000"


# =========================================================
# 3. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #111c32;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #60a5fa;
    margin-bottom: 0px;
}

.subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 25px;
}

.section-title {
    color: #60a5fa;
    font-size: 23px;
    font-weight: 700;
    margin-top: 20px;
}

.price-result {
    background: linear-gradient(135deg, #123b68, #164e8a);
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #3b82f6;
}

.price-result h2 {
    color: #bfdbfe;
    font-size: 20px;
}

.price-result h1 {
    color: white;
    font-size: 42px;
    font-weight: 800;
}

div[data-testid="stMetric"] {
    background-color: #16243d;
    border: 1px solid #263956;
    padding: 15px;
    border-radius: 12px;
}

.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 20px;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


df = load_data()


# =========================================================
# 5. LOAD MODEL FEATURE IMPORTANCE
# =========================================================

@st.cache_data
def get_model_feature_importance():

    if not MODEL_PATH.exists():
        return None

    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    rf_model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "encoded_feature": feature_names,
        "importance": importances
    })

    original_features = [
        "car_age",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "seats",
        "brand",
        "car_model",
        "variant_details",
        "fuel",
        "seller_type",
        "transmission",
        "owner"
    ]

    def get_original_feature(encoded_name):

        name = encoded_name.replace("num__", "")
        name = name.replace("cat__", "")

        for feature in original_features:

            if (
                name == feature
                or name.startswith(feature + "_")
            ):
                return feature

        return name

    importance_df["feature"] = (
        importance_df["encoded_feature"]
        .apply(get_original_feature)
    )

    result = (
        importance_df
        .groupby("feature")["importance"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    result["importance_percent"] = (
        result["importance"] * 100
    )

    return result


# =========================================================
# 6. LOAD ALGORITHM RESULTS
# =========================================================

@st.cache_data
def load_test_results():

    if not TEST_RESULTS_PATH.exists():
        return None

    return pd.read_csv(TEST_RESULTS_PATH)


@st.cache_data
def load_cv_results():

    if not CV_RESULTS_PATH.exists():
        return None

    return pd.read_csv(CV_RESULTS_PATH)


# =========================================================
# 7. DATASET OPTIONS
# =========================================================

if df is not None:

    brands = sorted(
        df["brand"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    brands = [
        "Maruti",
        "Hyundai",
        "Tata",
        "Honda"
    ]


DEFAULT_FUELS = [
    "Diesel",
    "Petrol",
    "CNG",
    "LPG",
    "Electric"
]

DEFAULT_SELLERS = [
    "Individual",
    "Dealer",
    "Trustmark Dealer"
]

DEFAULT_TRANSMISSIONS = [
    "Manual",
    "Automatic"
]

DEFAULT_OWNERS = [
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth & Above Owner",
    "Test Drive Car"
]


# =========================================================
# 8. CAR MODEL AND VARIANT OPTIONS
# =========================================================

def get_models(brand):

    if df is None:
        return ["Other / not listed"]

    filtered = df[
        df["brand"].astype(str) == brand
    ]

    options = sorted(
        filtered["car_model"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return options if options else ["Other / not listed"]


def get_variants(brand, car_model):

    if df is None:
        return ["Other / not listed"]

    filtered = df[
        (df["brand"].astype(str) == brand) &
        (df["car_model"].astype(str) == car_model)
    ]

    options = sorted(
        filtered["variant_details"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return options if options else ["Other / not listed"]


# =========================================================
# 9. CAR INPUT FORM
# =========================================================

def render_car_inputs(prefix):

    col1, col2 = st.columns(2)

    with col1:

        brand = st.selectbox(
            "Car Brand",
            brands,
            key=f"{prefix}_brand"
        )

    model_options = get_models(brand)

    with col2:

        car_model = st.selectbox(
            "Car Model",
            model_options,
            key=f"{prefix}_model"
        )

    variant_options = get_variants(
        brand,
        car_model
    )

    variant_details = st.selectbox(
        "Variant Details",
        variant_options,
        key=f"{prefix}_variant"
    )

    st.caption(
        "Select the closest matching model and variant "
        "available in the dataset."
    )

    st.markdown("#### Vehicle Specifications")

    col1, col2, col3 = st.columns(3)

    with col1:

        car_age = st.number_input(
            "Car Age (years)",
            min_value=0,
            max_value=50,
            value=6,
            step=1,
            key=f"{prefix}_age"
        )

    with col2:

        km_driven = st.number_input(
            "Kilometres Driven",
            min_value=0,
            value=60000,
            step=1000,
            key=f"{prefix}_km"
        )

    with col3:

        mileage = st.number_input(
            "Mileage (km/l)",
            min_value=0.1,
            max_value=100.0,
            value=20.0,
            step=0.1,
            key=f"{prefix}_mileage"
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        engine = st.number_input(
            "Engine (CC)",
            min_value=1,
            max_value=10000,
            value=1197,
            key=f"{prefix}_engine"
        )

    with col2:

        max_power = st.number_input(
            "Max Power (bhp)",
            min_value=0.1,
            max_value=2000.0,
            value=85.0,
            step=1.0,
            key=f"{prefix}_power"
        )

    with col3:

        seats = st.selectbox(
            "Number of Seats",
            [2, 4, 5, 6, 7, 8, 9, 10, 14],
            index=2,
            key=f"{prefix}_seats"
        )

    st.markdown("#### Additional Details")

    col1, col2 = st.columns(2)

    with col1:

        fuel = st.selectbox(
            "Fuel Type",
            DEFAULT_FUELS,
            key=f"{prefix}_fuel"
        )

        seller_type = st.selectbox(
            "Seller Type",
            DEFAULT_SELLERS,
            key=f"{prefix}_seller"
        )

    with col2:

        transmission = st.selectbox(
            "Transmission",
            DEFAULT_TRANSMISSIONS,
            key=f"{prefix}_transmission"
        )

        owner = st.selectbox(
            "Ownership",
            DEFAULT_OWNERS,
            key=f"{prefix}_owner"
        )

    return {
        "brand": brand,
        "car_model": car_model,
        "variant_details": variant_details,
        "car_age": car_age,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner
    }


# =========================================================
# 10. FASTAPI PREDICTION
# =========================================================

def predict_car(car_data):

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=car_data,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()

        try:

            error_detail = response.json().get(
                "detail",
                response.text
            )

        except ValueError:

            error_detail = response.text

        st.error(f"API Error: {error_detail}")

        return None

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI. "
            "Please start the backend."
        )

        return None

    except requests.exceptions.Timeout:

        st.error("Prediction request timed out.")

        return None

    except requests.exceptions.RequestException as e:

        st.error(f"Request failed: {e}")

        return None


# =========================================================
# 11. DISPLAY PREDICTION
# =========================================================

def display_prediction(result):

    price = float(result["predicted_price"])

    st.markdown(
        f"""
        <div class="price-result">
            <h2>Estimated Selling Price</h2>
            <h1>₹{price:,.0f}</h1>
            <p>Powered by CarWise V3.1 Random Forest</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "This is a machine-learning estimate, "
        "not a guaranteed market selling price."
    )


# =========================================================
# 12. PREDICTION EXPLANATION
# =========================================================

def show_feature_importance():

    st.markdown("---")

    st.markdown(
        "### 🧠 What Influences Car Prices?"
    )

    importance_df = get_model_feature_importance()

    if importance_df is None:

        st.warning(
            "Model feature importance is unavailable."
        )

        return

    st.bar_chart(
        importance_df.set_index("feature")[
            "importance_percent"
        ]
    )

    st.markdown("#### Feature Importance Ranking")

    display_df = importance_df.copy()

    display_df["importance_percent"] = (
        display_df["importance_percent"].round(2)
    )

    st.dataframe(
        display_df[
            ["feature", "importance_percent"]
        ].rename(columns={
            "feature": "Feature",
            "importance_percent": "Importance (%)"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These are global Random Forest feature importances. "
        "They describe overall model behaviour, not the exact "
        "reason behind an individual prediction. "
        "They should not be interpreted as causal effects."
    )


# =========================================================
# 13. HISTORICAL MARKET COMPARISON
# =========================================================

def show_market_comparison(car_data, predicted_price):

    st.markdown("---")

    st.markdown(
        "### 📊 Similar Cars Market Comparison"
    )

    if df is None:

        st.warning("Market dataset unavailable.")
        return

    similar = df[
        (df["brand"].astype(str) == car_data["brand"]) &
        (df["car_model"].astype(str) == car_data["car_model"])
    ].copy()

    if similar.empty:

        st.info("No matching cars found in the dataset.")
        return

    similar = similar[
        (similar["car_age"] >= car_data["car_age"] - 2) &
        (similar["car_age"] <= car_data["car_age"] + 2)
    ]

    if similar.empty:

        st.info(
            "No similar-age cars found in the historical dataset."
        )
        return

    market_median = float(
        similar["selling_price"].median()
    )

    market_mean = float(
        similar["selling_price"].mean()
    )

    difference = predicted_price - market_median

    percentage_difference = (
        difference / market_median * 100
        if market_median > 0 else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Price",
        f"₹{predicted_price:,.0f}"
    )

    col2.metric(
        "Historical Median",
        f"₹{market_median:,.0f}"
    )

    col3.metric(
        "Similar Cars Found",
        f"{len(similar):,}"
    )

    if difference > 0:

        st.info(
            f"Predicted price is {abs(percentage_difference):.1f}% "
            "above the historical similar-car median."
        )

    elif difference < 0:

        st.info(
            f"Predicted price is {abs(percentage_difference):.1f}% "
            "below the historical similar-car median."
        )

    else:

        st.success(
            "Predicted price matches the historical median."
        )

    comparison = pd.DataFrame({
        "Price Type": [
            "Predicted Price",
            "Historical Median",
            "Historical Mean"
        ],
        "Price (INR)": [
            predicted_price,
            market_median,
            market_mean
        ]
    })

    st.markdown("#### Price Comparison")

    st.bar_chart(
        comparison.set_index("Price Type")
    )

    st.markdown("#### Similar Historical Records")

    st.dataframe(
        similar[
            [
                "brand",
                "car_model",
                "variant_details",
                "car_age",
                "km_driven",
                "selling_price"
            ]
        ].sort_values(
            "selling_price",
            ascending=False
        ).head(10),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "This comparison uses historical CarDekho records, "
        "not live market listings. Similarity is based on "
        "brand, model and a +/- 2-year age range."
    )


# =========================================================
# 14. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        "<h1 style='color:#60a5fa;'>🚗 CarWise India</h1>",
        unsafe_allow_html=True
    )

    st.caption("Smart Used Car Price Prediction")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Price Estimator",
            "Market Analysis",
            "Compare Cars & Buying Insights",
            "Model Performance Dashboard"
        ]
    )

    st.divider()

    st.markdown("### Model Information")

    st.write("**Version:** V3.1")
    st.write("**Algorithm:** Random Forest")
    st.write("**Training records:** 6,926")
    st.write("**Input features:** 13")

    st.divider()

    if st.button("Check API Status"):

        try:

            response = requests.get(
                f"{API_URL}/health",
                timeout=5
            )

            if response.status_code == 200:
                st.success("API Connected")
            else:
                st.error("API is not healthy")

        except requests.exceptions.RequestException:
            st.error("API Offline")


# =========================================================
# 15. PAGE 1 - PRICE ESTIMATOR
# =========================================================

if page == "Price Estimator":

    st.markdown(
        '<div class="main-title">Car Price Estimator</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Estimate the resale value of your used car using '
        'our V3.1 Machine Learning model.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Enter Car Details</div>',
        unsafe_allow_html=True
    )

    with st.form("price_estimator_form"):

        car_data = render_car_inputs("estimate")

        submitted = st.form_submit_button(
            "Predict Car Price",
            use_container_width=True
        )

    if submitted:

        with st.spinner("Analyzing vehicle details..."):

            result = predict_car(car_data)

        if result and result.get("success"):

            st.success("Prediction generated successfully!")

            predicted_price = float(
                result["predicted_price"]
            )

            display_prediction(result)

            show_feature_importance()

            show_market_comparison(
                car_data,
                predicted_price
            )


# =========================================================
# 16. PAGE 2 - MARKET ANALYSIS
# =========================================================

elif page == "Market Analysis":

    st.markdown(
        '<div class="main-title">Used Car Market Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore brands, prices, vehicle specifications '
        'and historical market trends.'
        '</div>',
        unsafe_allow_html=True
    )

    if df is None:

        st.error("V3.1 dataset not found.")

    else:

        st.markdown("### Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Cars", f"{len(df):,}")

        col2.metric(
            "Unique Brands",
            df["brand"].nunique()
        )

        col3.metric(
            "Unique Models",
            df["car_model"].nunique()
        )

        col4.metric(
            "Average Price",
            f"₹{df['selling_price'].mean():,.0f}"
        )

        st.markdown("---")

        st.markdown("### Price Distribution")

        st.bar_chart(
            df["selling_price"]
            .value_counts(bins=20)
            .sort_index()
        )

        st.markdown("### Brand-wise Analysis")

        brand_analysis = (
            df.groupby("brand")["selling_price"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Average Price by Brand")

            st.bar_chart(
                brand_analysis["mean"].head(15)
            )

        with col2:

            st.markdown("#### Cars by Brand")

            st.bar_chart(
                brand_analysis["count"]
                .sort_values(ascending=False)
                .head(15)
            )

        st.markdown("### Model-wise Analysis")

        model_analysis = (
            df.groupby("car_model")["selling_price"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )

        st.dataframe(
            model_analysis.head(20),
            use_container_width=True
        )

        st.markdown("### Fuel Type Analysis")

        fuel_analysis = (
            df.groupby("fuel")["selling_price"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Average Price by Fuel")

            st.bar_chart(fuel_analysis["mean"])

        with col2:

            st.markdown("#### Vehicle Count")

            st.bar_chart(fuel_analysis["count"])

        st.markdown("### Transmission Analysis")

        transmission_analysis = (
            df.groupby("transmission")["selling_price"]
            .agg(["mean", "count"])
        )

        st.dataframe(
            transmission_analysis,
            use_container_width=True
        )

        st.markdown("### Car Age vs Selling Price")

        age_analysis = (
            df.groupby("car_age")["selling_price"]
            .mean()
            .sort_index()
        )

        st.line_chart(age_analysis)

        st.markdown("### Kilometres Driven vs Selling Price")

        scatter_data = df[
            ["km_driven", "selling_price"]
        ].dropna()

        st.scatter_chart(
            scatter_data,
            x="km_driven",
            y="selling_price"
        )

        st.markdown("### Statistical Summary")

        st.dataframe(
            df[
                [
                    "car_age",
                    "km_driven",
                    "mileage",
                    "engine",
                    "max_power",
                    "seats",
                    "selling_price"
                ]
            ].describe(),
            use_container_width=True
        )


# =========================================================
# 17. PAGE 3 - COMPARE CARS
# =========================================================

elif page == "Compare Cars & Buying Insights":

    st.markdown(
        '<div class="main-title">Compare Cars</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Compare two used cars using our V3.1 '
        'Random Forest model.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("compare_cars_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 🚗 Car A")

            car_a = render_car_inputs("compare_a")

        with col2:

            st.markdown("### 🚙 Car B")

            car_b = render_car_inputs("compare_b")

        compare_button = st.form_submit_button(
            "Compare Both Cars",
            use_container_width=True
        )

    if compare_button:

        with st.spinner("Calculating both car prices..."):

            result_a = predict_car(car_a)
            result_b = predict_car(car_b)

        if (
            result_a
            and result_b
            and result_a.get("success")
            and result_b.get("success")
        ):

            price_a = float(
                result_a["predicted_price"]
            )

            price_b = float(
                result_b["predicted_price"]
            )

            st.markdown("---")

            st.markdown("### Comparison Results")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    f"""
                    <div class="price-result">
                        <h2>Car A</h2>
                        <h1>₹{price_a:,.0f}</h1>
                        <p>{car_a["brand"]} {car_a["car_model"]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    f"""
                    <div class="price-result">
                        <h2>Car B</h2>
                        <h1>₹{price_b:,.0f}</h1>
                        <p>{car_b["brand"]} {car_b["car_model"]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            difference = abs(price_a - price_b)

            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Car A Price",
                f"₹{price_a:,.0f}"
            )

            col2.metric(
                "Car B Price",
                f"₹{price_b:,.0f}"
            )

            col3.metric(
                "Price Difference",
                f"₹{difference:,.0f}"
            )

            st.markdown("### Buying Insights")

            if price_a > price_b:

                st.success(
                    f"Car A has a higher predicted resale "
                    f"value by ₹{difference:,.0f}."
                )

            elif price_b > price_a:

                st.success(
                    f"Car B has a higher predicted resale "
                    f"value by ₹{difference:,.0f}."
                )

            else:

                st.info(
                    "Both cars have approximately the same "
                    "predicted resale value."
                )

            st.caption(
                "A higher predicted resale value does not "
                "automatically mean a better purchase. "
                "Consider vehicle condition, service history, "
                "accident history and ownership costs."
            )

            show_market_comparison(car_a, price_a)

            show_market_comparison(car_b, price_b)


# =========================================================
# 18. PAGE 4 - MODEL PERFORMANCE DASHBOARD
# =========================================================

elif page == "Model Performance Dashboard":

    st.markdown(
        '<div class="main-title">Model Performance Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Evaluate machine learning performance, '
        'compare algorithms and inspect validation results.'
        '</div>',
        unsafe_allow_html=True
    )

    test_results = load_test_results()
    cv_results = load_cv_results()

    if test_results is None:

        st.error(
            "Test results file not found: "
            "artifacts/v3_1_algorithm_test_results.csv"
        )

    else:

        required_columns = [
            "Algorithm",
            "MAE",
            "RMSE",
            "R2"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in test_results.columns
        ]

        if missing_columns:

            st.error(
                f"Missing columns in test results: "
                f"{missing_columns}"
            )

        else:

            rf_result = test_results[
                test_results["Algorithm"].astype(str)
                .str.contains("Random Forest", case=False)
            ]

            st.markdown("### Final Model Performance")

            if not rf_result.empty:

                rf_result = rf_result.iloc[0]

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Test MAE",
                    f"₹{rf_result['MAE']:,.0f}"
                )

                col2.metric(
                    "Test RMSE",
                    f"₹{rf_result['RMSE']:,.0f}"
                )

                col3.metric(
                    "R² Score",
                    f"{rf_result['R2']:.4f}"
                )

                col4.metric(
                    "Test Records",
                    "1,385"
                )

                st.success(
                    "Selected algorithm: Random Forest | "
                    "Primary selection metric: MAE"
                )

            else:

                st.warning(
                    "Random Forest results not found."
                )

            st.markdown("---")

            st.markdown("### Algorithm Comparison")

            comparison = test_results.sort_values("MAE")

            formatted_comparison = comparison[
                required_columns
            ].copy()

            formatted_comparison["MAE"] = (
                formatted_comparison["MAE"]
                .map(lambda x: f"₹{x:,.2f}")
            )

            formatted_comparison["RMSE"] = (
                formatted_comparison["RMSE"]
                .map(lambda x: f"₹{x:,.2f}")
            )

            formatted_comparison["R2"] = (
                formatted_comparison["R2"]
                .map(lambda x: f"{x:.4f}")
            )

            st.dataframe(
                formatted_comparison,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("#### MAE Comparison")

            st.bar_chart(
                comparison.set_index("Algorithm")["MAE"]
            )

            st.markdown("#### RMSE Comparison")

            st.bar_chart(
                comparison.set_index("Algorithm")["RMSE"]
            )

            st.markdown("#### R² Comparison")

            st.bar_chart(
                comparison.set_index("Algorithm")["R2"]
            )

            st.markdown("---")

            st.markdown("### Grouped Cross-Validation Results")

            if cv_results is not None:

                st.dataframe(
                    cv_results,
                    use_container_width=True,
                    hide_index=True
                )

                if (
                    "Algorithm" in cv_results.columns
                    and "Best_CV_MAE" in cv_results.columns
                ):

                    st.markdown(
                        "#### Cross-Validation MAE"
                    )

                    st.bar_chart(
                        cv_results.set_index("Algorithm")[
                            "Best_CV_MAE"
                        ]
                    )

            else:

                st.warning(
                    "Cross-validation results CSV not found."
                )

            st.markdown("---")

            st.markdown("### Understanding Evaluation Metrics")

            with st.expander("Mean Absolute Error (MAE)"):

                st.write(
                    "MAE measures the average absolute difference "
                    "between actual and predicted prices. "
                    "Lower MAE is better."
                )

            with st.expander("Root Mean Squared Error (RMSE)"):

                st.write(
                    "RMSE penalizes larger prediction errors more "
                    "strongly than MAE. Lower RMSE is better."
                )

            with st.expander("R² Score"):

                st.write(
                    "R² measures how much of the variation in "
                    "selling prices is explained by the model. "
                    "A higher score is generally better."
                )

            st.markdown("### Experiment Conclusion")

            st.write(
                "Random Forest was selected because it achieved "
                "the lowest MAE on the untouched test set. "
                "Hist Gradient Boosting achieved the lowest RMSE "
                "and highest R². Extra Trees achieved the lowest "
                "grouped cross-validation MAE."
            )

            st.caption(
                "Test metrics come from the reserved 1,385-record "
                "holdout set. The final production model was "
                "subsequently trained on all 6,926 records. "
                "Do not interpret training performance as "
                "independent test performance."
            )


# =========================================================
# 19. FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        padding:15px;
    ">
        <b>CarWise India</b> | V3.1 Random Forest Model<br>
        Used Car Price Prediction using Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
