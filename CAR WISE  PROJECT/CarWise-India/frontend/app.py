"""
CarWise India V4 — Deployment-oriented Streamlit frontend
---------------------------------------------------------
Run from the project root:
    python -m streamlit run frontend/app.py

Required project files:
    data/processed/carwise_v3_1_model_data.csv
    artifacts/v3_1_random_forest_final.joblib
    artifacts/v3_1_algorithm_test_results.csv
    artifacts/v3_1_algorithm_cv_results.csv
    backend/main.py  (FastAPI service, started separately)

Install:
    python -m pip install streamlit pandas numpy requests joblib plotly reportlab folium streamlit-folium
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from io import BytesIO
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ---------------------------------------------------------------------
# 1. APP CONFIGURATION AND PATHS
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="CarWise India | Auto Market Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "carwise_v3_1_model_data.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "v3_1_random_forest_final.joblib"
TEST_RESULTS_PATH = BASE_DIR / "artifacts" / "v3_1_algorithm_test_results.csv"
CV_RESULTS_PATH = BASE_DIR / "artifacts" / "v3_1_algorithm_cv_results.csv"

try:
    _configured_api_url = st.secrets.get("API_URL", "http://127.0.0.1:8000")
except Exception:
    _configured_api_url = "http://127.0.0.1:8000"
API_URL = str(__import__("os").getenv("API_URL", _configured_api_url)).rstrip("/")
REFERENCE_YEAR = 2020
EXPECTED_FEATURES = [
    "car_age", "km_driven", "mileage", "engine", "max_power", "seats",
    "brand", "car_model", "variant_details", "fuel", "seller_type",
    "transmission", "owner",
]
NUMERIC_FEATURES = [
    "car_age", "km_driven", "mileage", "engine", "max_power", "seats"
]
CATEGORICAL_FEATURES = [
    "brand", "car_model", "variant_details", "fuel", "seller_type",
    "transmission", "owner"
]

# CarWise visual system — editorial automotive journal
# Warm paper, deep ink, muted olive and a restrained burnt-orange accent.
ACCENT_PRIMARY = "#B65E3C"
ACCENT_SECONDARY = "#8B9272"
ACCENT_CYAN = "#718477"
BG_DARK = "#F4F1E9"
PANEL_BG = "#FFFEFA"
TEXT_COLOR = "#242923"
BORDER_COLOR = "#DED9CE"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');

    :root {
        --paper: #F4F1E9;
        --paper-light: #FBFAF6;
        --ink: #242923;
        --muted: #73766D;
        --line: #DED9CE;
        --olive: #718477;
        --rust: #B65E3C;
        --soft-rust: #F2E4DC;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }
    .stApp {
        background: var(--paper);
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(244,241,233,.94);
        border-bottom: 1px solid var(--line);
    }
    [data-testid="stToolbar"] { right: 1.5rem; }

    .block-container {
        max-width: 1380px;
        padding-top: 2.1rem;
        padding-bottom: 3.5rem;
    }

    /* Sidebar: quiet, functional, closer to a publication index */
    [data-testid="stSidebar"] {
        background: #EAE7DE;
        border-right: 1px solid #D8D3C8;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    [data-testid="stSidebar"] * { color: var(--ink); }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        font-size: .68rem;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-weight: 700;
        color: #77796F;
        margin-bottom: .6rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: .18rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: .55rem .72rem;
        border-radius: 4px;
        border: 1px solid transparent;
        background: transparent;
        transition: background .15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #F5F2EA;
        border-color: #D9D3C7;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #F9F7F1;
        border-color: #D8D2C6;
        box-shadow: inset 3px 0 0 var(--rust);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: .9rem;
        font-weight: 600;
    }

    /* Editorial page heading: no gradient, glow or oversized app-like banner */
    .cw-hero {
        background: transparent;
        border: 0;
        border-bottom: 1px solid var(--line);
        border-radius: 0;
        box-shadow: none;
        padding: .35rem 0 1.55rem;
        margin: 0 0 1.8rem;
        backdrop-filter: none;
    }
    .cw-eyebrow {
        color: var(--rust);
        font-size: .67rem;
        font-weight: 700;
        letter-spacing: .17em;
        text-transform: uppercase;
        margin-bottom: .65rem;
    }
    .cw-title {
        color: var(--ink);
        font-family: 'Newsreader', Georgia, serif;
        font-size: clamp(2.25rem, 4vw, 3.45rem);
        font-weight: 500;
        letter-spacing: -.035em;
        line-height: 1.02;
        margin: 0 0 .7rem;
        background: none;
        -webkit-background-clip: initial;
        -webkit-text-fill-color: currentColor;
    }
    .cw-subtitle {
        color: var(--muted);
        font-size: .96rem;
        max-width: 760px;
        line-height: 1.7;
    }

    /* Data cards: flat surfaces and thin rules */
    .cw-card {
        background: var(--paper-light);
        border: 1px solid var(--line);
        border-top: 1px solid var(--line);
        border-radius: 5px;
        padding: 1.05rem 1.15rem 1.1rem;
        box-shadow: 0 2px 8px rgba(40,38,30,.025);
        min-height: 118px;
        transition: border-color .15s ease, transform .15s ease;
    }
    .cw-card:hover {
        transform: translateY(-1px);
        border-color: #BEB7A9;
        box-shadow: 0 4px 12px rgba(40,38,30,.045);
    }
    .cw-card-label {
        color: var(--muted);
        font-size: .67rem;
        font-weight: 700;
        letter-spacing: .11em;
        text-transform: uppercase;
    }
    .cw-card-value {
        color: var(--ink);
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.05rem;
        font-weight: 500;
        line-height: 1.15;
        margin-top: .5rem;
        letter-spacing: -.025em;
    }
    .cw-card-note {
        color: #85867D;
        font-size: .74rem;
        line-height: 1.45;
        margin-top: .35rem;
    }

    .cw-result {
        background: #28342D;
        border: 0;
        border-left: 4px solid var(--rust);
        border-radius: 4px;
        padding: 1.65rem 1.8rem;
        margin: 1rem 0;
        box-shadow: none;
    }
    .cw-result-label {
        color: #D7B7A6;
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .cw-result-price {
        color: #FFFDF7;
        font-family: 'Newsreader', Georgia, serif;
        font-size: clamp(2.6rem, 5vw, 3.8rem);
        font-weight: 500;
        letter-spacing: -.035em;
        margin: .3rem 0;
        text-shadow: none;
    }

    /* Section labels and notes */
    .cw-section {
        color: var(--ink);
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.65rem;
        font-weight: 500;
        margin: 2rem 0 .9rem;
        letter-spacing: -.02em;
        display: block;
    }
    .cw-section::before { display: none; }
    .cw-note {
        border: 1px solid #E1D7C9;
        border-left: 3px solid var(--olive);
        padding: .85rem 1rem;
        background: #F8F5ED;
        border-radius: 3px;
        color: #62665D;
        font-size: .84rem;
        line-height: 1.65;
        margin: .8rem 0 1rem;
    }
    .cw-insight-title {
        color: #77796F;
        font-size: .64rem;
        font-weight: 700;
        letter-spacing: .15em;
        margin: .75rem 0 .35rem;
        text-transform: uppercase;
    }

    /* Native Streamlit components */
    h1, h2, h3, h4 {
        color: var(--ink);
        font-family: 'Newsreader', Georgia, serif !important;
        font-weight: 500 !important;
        letter-spacing: -.02em;
    }
    p, li, label, .stMarkdown { color: var(--ink); }
    .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted); }

    /* Keep the select control and its popup menu consistent with the light theme. */
    [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSelectbox"] [role="combobox"] {
        background-color: #FFFEFA !important;
        color: #242923 !important;
        border-color: #D8D2C7 !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] *,
    [data-testid="stSelectbox"] [role="combobox"] * {
        color: #242923 !important;
        -webkit-text-fill-color: #242923 !important;
        opacity: 1 !important;
    }

    /* Streamlit renders dropdown options in a popup portal outside the widget. */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"],
    [role="listbox"] {
        background-color: #FFFEFA !important;
        color: #242923 !important;
        border-color: #DED9CE !important;
    }
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="menu"] [role="option"],
    [role="listbox"] [role="option"] {
        background-color: #FFFEFA !important;
        color: #242923 !important;
        -webkit-text-fill-color: #242923 !important;
        opacity: 1 !important;
    }
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="menu"] [role="option"]:hover,
    [role="listbox"] [role="option"]:hover {
        background-color: #EAE7DE !important;
        color: #242923 !important;
    }
    [data-baseweb="popover"] [aria-selected="true"],
    [data-baseweb="menu"] [aria-selected="true"],
    [role="listbox"] [aria-selected="true"] {
        background-color: #E2E7DF !important;
        color: #28342D !important;
        font-weight: 700 !important;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stMultiSelect"] > div > div {
        background: #FFFEFA !important;
        color: var(--ink) !important;
        border: 1px solid #D8D2C7 !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--olive) !important;
        box-shadow: 0 0 0 1px var(--olive) !important;
    }
    [data-testid="stForm"] {
        background: #F9F7F1;
        border: 1px solid var(--line);
        border-radius: 5px;
        padding: 1rem 1.2rem;
    }
    .stButton > button, .stFormSubmitButton > button,
    .stDownloadButton > button {
        background: #34463B !important;
        color: #FFFDF7 !important;
        border: 1px solid #34463B !important;
        border-radius: 4px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: .86rem !important;
        font-weight: 600 !important;
        letter-spacing: .01em !important;
        padding: .58rem 1.05rem !important;
        box-shadow: none !important;
        transition: background .15s ease, border-color .15s ease !important;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {
        background: #25362C !important;
        border-color: #25362C !important;
        transform: none !important;
        box-shadow: none !important;
    }
    .stButton > button:focus, .stFormSubmitButton > button:focus {
        box-shadow: 0 0 0 2px #B9C3B7 !important;
    }

    [data-testid="stTabs"] button {
        color: #77796F;
        font-size: .85rem;
        font-weight: 600;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--ink);
        border-bottom-color: var(--rust);
    }
    [data-testid="stExpander"] {
        background: #F9F7F1;
        border: 1px solid var(--line);
        border-radius: 4px;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 4px;
        overflow: hidden;
    }
    [data-testid="stMetric"] {
        background: #F9F7F1;
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: .8rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Newsreader', Georgia, serif;
    }
    hr { border-color: var(--line); }
    footer { visibility: hidden; }

    /* Keep native Streamlit controls above decorative containers and clickable. */
    [data-testid="stSelectbox"], [data-testid="stNumberInput"],
    [data-testid="stTextInput"], [data-testid="stRadio"],
    [data-testid="stButton"], [data-testid="stFormSubmitButton"],
    [data-testid="stDownloadButton"], [data-testid="stCheckbox"],
    [data-testid="stMultiSelect"] {
        position: relative;
        z-index: 2;
        pointer-events: auto !important;
    }
    [data-testid="stSelectbox"] *, [data-testid="stNumberInput"] *,
    [data-testid="stTextInput"] *, [data-testid="stRadio"] *,
    [data-testid="stButton"] *, [data-testid="stCheckbox"] *,
    [data-testid="stMultiSelect"] * {
        pointer-events: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 2. DATA ACCESS AND VALIDATION
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataset(path_string: str, modified_time: float) -> pd.DataFrame:
    data = pd.read_csv(path_string)
    required = set(EXPECTED_FEATURES + ["selling_price"])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    for col in NUMERIC_FEATURES + ["selling_price"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["selling_price"]).copy()
    data = data[data["selling_price"] > 0]
    return data


@st.cache_resource(show_spinner="Loading the production model...")
def load_model(path_string: str, modified_time: float):
    return joblib.load(path_string)


@st.cache_data(show_spinner=False)
def load_results(path_string: str, modified_time: float) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path_string)
    except Exception:
        return None


def file_stamp(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def safe_dataset() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    try:
        return load_dataset(str(DATA_PATH), file_stamp(DATA_PATH))
    except Exception as exc:
        st.error(f"Dataset could not be loaded: {exc}")
        return None


def money(value: float) -> str:
    return f"₹{value:,.0f}"


def short_money(value: float) -> str:
    value = float(value)
    if value >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} Cr"
    if value >= 100_000:
        return f"₹{value / 100_000:.2f} L"
    if value >= 1_000:
        return f"₹{value / 1_000:.0f} K"
    return f"₹{value:,.0f}"


GRAPH_NOTES = {
    "Distribution of historical selling prices": [
        "Check where most listings are concentrated along the price axis.",
        "A long right tail indicates a smaller number of high-priced vehicles.",
        "The box marker summarises the median and spread of the distribution.",
    ],
    "Most represented brands": [
        "Bars show listing frequency, not brand popularity in the entire Indian market.",
        "Longer bars indicate stronger representation in this dataset.",
        "Brand counts help identify where the dataset has more training examples.",
    ],
    "Records by price segment": [
        "Compares how many observations fall into each predefined price band.",
        "Uneven segment sizes indicate that the dataset is not price-balanced.",
        "Smaller premium and luxury groups provide less evidence for those segments.",
    ],
    "Median price by segment": [
        "Median price rises across the budget, mid-range, premium and luxury bands by construction.",
        "The median is less affected by extreme listings than the mean.",
        "Segment boundaries are analytical categories, not official market classifications.",
    ],
    "Historical prices for same brand/model and ±2-year age": [
        "The bars show historical prices for the selected brand, model family and age window.",
        "The dashed marker indicates the model's estimate for the entered specifications.",
        "This is a broad historical comparison, not a live or condition-matched valuation.",
    ],
    "Selling price distribution": [
        "Shows the spread of prices in the currently selected filters.",
        "Use the central mass to understand the typical price range.",
        "Extreme values can stretch the axis and make most records appear compressed.",
    ],
    "Price distribution by fuel type": [
        "Each box compares the median and interquartile price range for a fuel category.",
        "Points beyond the whiskers represent potential price outliers.",
        "Differences may also reflect brand, age and vehicle-class composition.",
    ],
    "Share of filtered records by price segment": [
        "Each slice represents a segment's share of the filtered observations.",
        "Larger slices indicate stronger representation in the selected subset.",
        "The chart describes this dataset, not national used-car market share.",
    ],
    "Brands ranked by median price (minimum 5 records)": [
        "Brands are ordered by their median observed selling price.",
        "The five-record minimum reduces rankings based on extremely small samples.",
        "A higher median does not mean every model from that brand is expensive.",
    ],
    "Brand volume vs median price": [
        "Each point represents a brand; its position compares record count with median price.",
        "Point size reflects the number of observations for that brand.",
        "The pattern is descriptive and does not establish a brand-price causal effect.",
    ],
    "Manual vs automatic price distribution": [
        "Compares the median and spread of prices for each transmission category.",
        "Outlier points indicate observations far from the central distribution.",
        "Vehicle age, brand and engine class can influence this comparison.",
    ],
    "Age vs price: median and interquartile range": [
        "The line tracks median selling price at each recorded car age.",
        "The shaded band represents the middle 50% of prices (25th–75th percentiles).",
        "Age uses the project's fixed 2020 reference-year convention.",
    ],
    "Median price by age and fuel type": [
        "Separate lines compare median prices across fuel categories as recorded age changes.",
        "Gaps between lines describe differences in the filtered historical observations.",
        "Category counts can vary by age, so interpret sparse portions cautiously.",
    ],
    "Kilometres driven vs selling price": [
        "Each point represents a vehicle's recorded distance travelled and selling price.",
        "Colour encodes maximum power, adding a second specification dimension.",
        "The scatter reveals association and spread; it does not isolate mileage's effect.",
    ],
    "Maximum power vs selling price": [
        "Compares recorded engine power with selling price across sampled vehicles.",
        "Colour represents engine capacity, helping reveal overlapping vehicle profiles.",
        "Brand, model, age and condition also contribute to observed price differences.",
    ],
    "Mileage vs selling price": [
        "Each point compares fuel efficiency with the recorded selling price.",
        "Colour distinguishes fuel categories, which may have different vehicle mixes.",
        "A visible trend is an association, not proof that mileage determines price.",
    ],
    "Spearman rank-correlation heatmap": [
        "Values range from −1 to +1 and describe monotonic relationships between variables.",
        "Positive values indicate variables tend to increase together; negative values indicate opposite ranking.",
        "Correlation is not causation and may be affected by other vehicle characteristics.",
    ],
    "Predicted resale value comparison": [
        "The bars compare the model estimates for the two entered vehicles.",
        "The taller bar indicates the higher predicted resale value, not necessarily the better purchase.",
        "Condition, service history, accident records and ownership costs are not assessed.",
    ],
    "Holdout MAE (lower is better)": [
        "MAE is the average absolute prediction error, measured in rupees.",
        "A shorter bar indicates lower average error on the experiment holdout.",
        "MAE treats each absolute error equally, so inspect RMSE for large-error sensitivity.",
    ],
    "Holdout RMSE (lower is better)": [
        "RMSE is measured in rupees and penalises large errors more strongly than MAE.",
        "A shorter bar indicates lower squared-error magnitude on the holdout.",
        "Compare it alongside MAE rather than using it as the only selection metric.",
    ],
    "Holdout R² (higher is better)": [
        "R² compares model fit with a baseline that predicts the target mean.",
        "Values closer to 1 indicate stronger fit on this holdout.",
        "A high R² does not guarantee small errors for every price segment.",
    ],
    "Grouped cross-validation MAE (lower is better)": [
        "Bars compare mean absolute error across the grouped validation procedure.",
        "Lower values indicate better average validation performance.",
        "Cross-validation results estimate stability across folds; they are not a separate final test.",
    ],
    "Aggregated Random Forest feature importance": [
        "Importance is aggregated back to the original input-feature level.",
        "Larger values indicate greater contribution to the forest's impurity-based split criteria.",
        "Importance is global, model-specific and not a causal explanation for an individual estimate.",
    ],
    "Missing values by feature": [
        "Bars show the percentage of records missing each feature.",
        "Features with more missing values rely more heavily on pipeline imputation.",
        "The target and input completeness should be checked before interpreting model results.",
    ],
}


def show_graph_notes(fig) -> None:
    title_obj = getattr(getattr(fig, "layout", None), "title", None)
    title = getattr(title_obj, "text", "") if title_obj is not None else ""
    notes = GRAPH_NOTES.get(title)
    if not notes:
        notes = [
            "Read the axis labels and units before comparing values.",
            "Look for the main pattern, spread and any unusually distant observations.",
            "Treat this chart as descriptive evidence from the available historical dataset.",
        ]
    st.markdown('<div class="cw-insight-title">KEY TAKEAWAYS</div>', unsafe_allow_html=True)
    for note in notes:
        st.markdown(f"- {note}")


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<section class="cw-hero">'
        f'<div class="cw-eyebrow">{eyebrow}</div>'
        f'<div class="cw-title">{title}</div>'
        f'<div class="cw-subtitle">{subtitle}</div>'
        f'</section>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="cw-card">
          <div class="cw-card-label">{label}</div>
          <div class="cw-card-value">{value}</div>
          <div class="cw-card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_layout(fig, height: int = 390):
    """Apply the CarWise paper-and-ink chart style."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFEFA",
        plot_bgcolor="#FFFEFA",
        font=dict(color="#343A33", family="DM Sans, sans-serif", size=12),
        margin=dict(l=28, r=22, t=58, b=42),
        height=height,
        colorway=["#B65E3C", "#718477", "#C49A62", "#536B78", "#A58A78", "#8B9272"],
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=11, color="#62665D")
        ),
        hoverlabel=dict(
            bgcolor="#28342D", font_color="#FFFDF7", bordercolor="#28342D",
            font_family="DM Sans, sans-serif"
        ),
        title_font=dict(color="#242923", family="Newsreader, Georgia, serif", size=18),
        title_x=0.015,
        title_y=0.97,
    )
    fig.update_xaxes(
        gridcolor="#EEEAE1", zerolinecolor="#D8D2C7",
        linecolor="#D8D2C7", tickfont=dict(color="#73766D", size=10),
        title_font=dict(color="#62665D", size=11),
        showline=True, linewidth=1,
    )
    fig.update_yaxes(
        gridcolor="#EEEAE1", zerolinecolor="#D8D2C7",
        linecolor="#D8D2C7", tickfont=dict(color="#73766D", size=10),
        title_font=dict(color="#62665D", size=11),
        showline=False,
    )
    return fig



def show_empty_data_message():
    st.error(
        "The V3.1 dataset is unavailable or has an unexpected schema. "
        "Check data/processed/carwise_v3_1_model_data.csv."
    )


df = safe_dataset()


# ---------------------------------------------------------------------
# 3. API AND MODEL HELPERS
# ---------------------------------------------------------------------

def check_api() -> tuple[bool, str]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.ok:
            body = response.json()
            if body.get("model_loaded") is True:
                return True, "API online · model loaded"
            return False, "API responded, but model is not loaded"
        return False, f"API returned HTTP {response.status_code}"
    except requests.exceptions.RequestException:
        return False, "API offline · start FastAPI backend"


def predict_car(car_data: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=car_data,
            timeout=(5, 30),
        )
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(f"Prediction service error ({response.status_code}): {detail}")
            return None
        result = response.json()
        if not result.get("success") or "predicted_price" not in result:
            st.error("The prediction service returned an unexpected response.")
            return None
        price = float(result["predicted_price"])
        if not np.isfinite(price) or price <= 0:
            st.error("The prediction service returned an invalid price.")
            return None
        result["predicted_price"] = price
        return result
    except requests.exceptions.Timeout:
        st.error("The prediction request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI. Start the backend and retry.")
    except (requests.exceptions.RequestException, ValueError, TypeError) as exc:
        st.error(f"Prediction request failed: {exc}")
    return None


def get_feature_importance() -> pd.DataFrame | None:
    if not MODEL_PATH.exists():
        return None
    try:
        pipeline = load_model(str(MODEL_PATH), file_stamp(MODEL_PATH))
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]
        if not hasattr(estimator, "feature_importances_"):
            return None
        names = preprocessor.get_feature_names_out()
        values = estimator.feature_importances_
        if len(names) != len(values):
            return None
        encoded = pd.DataFrame({"encoded": names, "importance": values})
        def source_feature(name: str) -> str:
            clean = name.split("__", 1)[-1]
            for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
                if clean == feature or clean.startswith(feature + "_"):
                    return feature
            return clean
        encoded["feature"] = encoded["encoded"].map(source_feature)
        result = (
            encoded.groupby("feature", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=False)
        )
        result["importance_pct"] = result["importance"] * 100
        return result
    except Exception as exc:
        st.warning(f"Feature importance could not be loaded: {exc}")
        return None


def get_categories(data: pd.DataFrame, column: str, fallback: list[str]) -> list[str]:
    if data is None or column not in data:
        return fallback
    values = sorted(data[column].dropna().astype(str).unique().tolist())
    return values or fallback


def model_options(data: pd.DataFrame, brand: str) -> list[str]:
    vals = data.loc[data["brand"].astype(str).eq(brand), "car_model"]
    return sorted(vals.dropna().astype(str).unique().tolist()) or ["Other / not listed"]


def variant_options(data: pd.DataFrame, brand: str, model: str) -> list[str]:
    mask = data["brand"].astype(str).eq(brand) & data["car_model"].astype(str).eq(model)
    vals = data.loc[mask, "variant_details"]
    return sorted(vals.dropna().astype(str).unique().tolist()) or ["Other / not listed"]


def render_car_form(prefix: str, data: pd.DataFrame) -> dict[str, Any]:
    brands = get_categories(data, "brand", ["Maruti"])
    fuel_values = get_categories(data, "fuel", ["Diesel", "Petrol"])
    seller_values = get_categories(data, "seller_type", ["Individual"])
    transmission_values = get_categories(data, "transmission", ["Manual"])
    owner_values = get_categories(data, "owner", ["First Owner"])

    c1, c2 = st.columns(2)
    with c1:
        brand = st.selectbox("Brand", brands, key=f"{prefix}_brand")
    models = model_options(data, brand)
    with c2:
        car_model = st.selectbox("Model family", models, key=f"{prefix}_model_{brand}")
    variants = variant_options(data, brand, car_model)
    variant_details = st.selectbox("Variant / trim", variants, key=f"{prefix}_variant_{brand}_{car_model}")

    st.markdown("**Vehicle specifications**")
    c1, c2, c3 = st.columns(3)
    with c1:
        car_age = st.number_input(
            "Car age (years)", min_value=0, max_value=50, value=6,
            step=1, key=f"{prefix}_age",
            help="The model uses the project's 2020 reference year: enter 2020 minus manufacturing year.",
        )
    with c2:
        km_driven = st.number_input(
            "Kilometres driven", min_value=0, max_value=2_000_000,
            value=60_000, step=1_000, key=f"{prefix}_km",
        )
    with c3:
        mileage = st.number_input(
            "Mileage (km/l)", min_value=0.1, max_value=100.0,
            value=20.0, step=0.1, key=f"{prefix}_mileage",
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        engine = st.number_input(
            "Engine capacity (cc)", min_value=1, max_value=10_000,
            value=1197, step=1, key=f"{prefix}_engine",
        )
    with c2:
        max_power = st.number_input(
            "Maximum power (bhp)", min_value=0.1, max_value=2_000.0,
            value=85.0, step=0.5, key=f"{prefix}_power",
        )
    with c3:
        seats_options = sorted(set([2, 4, 5, 6, 7, 8, 9, 10, 14]))
        seats = st.selectbox("Seats", seats_options, index=2, key=f"{prefix}_seats")

    st.markdown("**Ownership and drivetrain**")
    c1, c2 = st.columns(2)
    with c1:
        fuel = st.selectbox("Fuel type", fuel_values, key=f"{prefix}_fuel")
        seller_type = st.selectbox("Seller type", seller_values, key=f"{prefix}_seller")
    with c2:
        transmission = st.selectbox("Transmission", transmission_values, key=f"{prefix}_transmission")
        owner = st.selectbox("Ownership", owner_values, key=f"{prefix}_owner")

    return {
        "brand": brand,
        "car_model": car_model,
        "variant_details": variant_details,
        "car_age": float(car_age),
        "km_driven": float(km_driven),
        "mileage": float(mileage),
        "engine": float(engine),
        "max_power": float(max_power),
        "seats": int(seats),
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }


def show_prediction(result: dict[str, Any], car_data: dict[str, Any]) -> None:
    price = result["predicted_price"]
    st.markdown(
        f"""
        <div class="cw-result">
          <div class="cw-result-label">ESTIMATED RESALE VALUE</div>
          <div class="cw-result-price">{money(price)}</div>
          <div style="color:#94A3B8;font-size:0.85rem;font-weight:500;">
            CarWise V3.1 · Random Forest Model Estimate · INR
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "This is a model estimate based on historical CarDekho records, "
        "not a live quote, inspection-based valuation, or guaranteed selling price."
    )
    st.write(
        f"**Vehicle:** {car_data['brand']} {car_data['car_model']} — "
        f"{car_data['variant_details']}"
    )


def historical_comparison(data: pd.DataFrame, car: dict[str, Any], predicted: float) -> pd.DataFrame:
    mask = (
        data["brand"].astype(str).eq(car["brand"])
        & data["car_model"].astype(str).eq(car["car_model"])
        & data["car_age"].between(car["car_age"] - 2, car["car_age"] + 2)
    )
    return data.loc[mask].copy()



def build_valuation_pdf(
    car: dict[str, Any],
    result: dict[str, Any],
    asking_price: float | None = None,
    matches: pd.DataFrame | None = None,
    feature_importance: pd.DataFrame | None = None,
) -> bytes | None:
    """Create a concise, two-page, user-specific valuation report."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, KeepTogether
        )
        from reportlab.graphics.shapes import Drawing, String
        from reportlab.graphics.charts.barcharts import HorizontalBarChart

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4, rightMargin=15*mm, leftMargin=15*mm,
            topMargin=13*mm, bottomMargin=13*mm
        )
        paper = colors.HexColor("#F4F1E9")
        ink = colors.HexColor("#28342D")
        olive = colors.HexColor("#718477")
        terracotta = colors.HexColor("#B65E3C")
        muted = colors.HexColor("#667067")
        line = colors.HexColor("#DED9CE")
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name="CWBrand", parent=styles["Title"], fontName="Helvetica-Bold",
            fontSize=19, leading=22, textColor=ink, alignment=0, spaceAfter=2
        ))
        styles.add(ParagraphStyle(
            name="CWSub", parent=styles["BodyText"], fontSize=8, leading=11,
            textColor=muted, spaceAfter=5
        ))
        styles.add(ParagraphStyle(
            name="CWSection", parent=styles["Heading2"], fontName="Helvetica-Bold",
            fontSize=11, leading=14, textColor=ink, spaceBefore=7, spaceAfter=5
        ))
        styles.add(ParagraphStyle(
            name="CWBody", parent=styles["BodyText"], fontSize=8.2,
            leading=11, textColor=ink
        ))
        styles.add(ParagraphStyle(
            name="CWNote", parent=styles["BodyText"], fontSize=7,
            leading=9, textColor=muted
        ))
        styles.add(ParagraphStyle(
            name="CWPrice", parent=styles["Title"], fontName="Helvetica-Bold",
            fontSize=25, leading=28, textColor=ink, alignment=0
        ))

        price = float(result["predicted_price"])
        vehicle = f"{car.get('brand','')} {car.get('car_model','')} {car.get('variant_details','')}".strip()
        story = [
            Paragraph("CARWISE INDIA", styles["CWBrand"]),
            Paragraph("VEHICLE VALUATION REPORT", styles["CWSection"]),
            Paragraph(
                f"{vehicle} &nbsp; | &nbsp; Generated {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
                styles["CWSub"]
            ),
        ]
        price_rows = [
            [Paragraph("CARWISE ESTIMATED VALUE", styles["CWNote"]),
             Paragraph("ASKING PRICE" if asking_price and asking_price > 0 else "VALUATION BASIS", styles["CWNote"])],
            [Paragraph(money(price), styles["CWPrice"]),
             Paragraph(money(asking_price) if asking_price and asking_price > 0 else "Historical ML estimate", styles["CWSection"])]
        ]
        price_table = Table(price_rows, colWidths=[88*mm, 88*mm], rowHeights=[8*mm, 16*mm])
        price_table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),paper), ("BOX",(0,0),(-1,-1),0.7,line),
            ("LINEBEFORE",(1,0),(1,-1),0.7,line), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING",(0,0),(-1,-1),8),
            ("TOPPADDING",(0,0),(-1,-1),4), ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ]))
        story += [price_table, Spacer(1, 3*mm)]

        if asking_price and asking_price > 0:
            diff = asking_price - price
            pct = diff / price * 100 if price else 0
            verdict = "Above model estimate" if pct > 5 else ("Below model estimate" if pct < -5 else "Close to model estimate")
            story.append(Paragraph(
                f"<b>Asking-price comparison:</b> {verdict}. The seller's asking price is "
                f"{money(abs(diff))} ({abs(pct):.1f}%) "
                f"{'higher' if diff > 0 else 'lower' if diff < 0 else 'the same as'} the model estimate. "
                "This is a comparison with the model output, not proof of a fair market price.",
                styles["CWBody"]
            ))

        story.append(Paragraph("Vehicle profile", styles["CWSection"]))
        labels = {
            "brand":"Brand", "car_model":"Model", "variant_details":"Variant",
            "car_age":"Car age (2020 reference)", "km_driven":"Kilometres driven",
            "mileage":"Mileage (km/l)", "engine":"Engine (cc)", "max_power":"Power (bhp)",
            "seats":"Seats", "fuel":"Fuel", "seller_type":"Seller type",
            "transmission":"Transmission", "owner":"Ownership"
        }
        details = [[labels.get(k,k), str(v)] for k,v in car.items()]
        detail_table = Table(details, colWidths=[43*mm, 133*mm], hAlign="LEFT")
        detail_table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(0,-1),paper), ("TEXTCOLOR",(0,0),(0,-1),ink),
            ("GRID",(0,0),(-1,-1),0.35,line), ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),7.5), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ROWBACKGROUNDS",(1,0),(1,-1),[colors.white,colors.HexColor("#FBFAF6")]),
            ("LEFTPADDING",(0,0),(-1,-1),6), ("RIGHTPADDING",(0,0),(-1,-1),6),
            ("TOPPADDING",(0,0),(-1,-1),3), ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ]))
        story.append(detail_table)

        story.append(Paragraph("What the historical data says", styles["CWSection"]))
        if matches is not None and not matches.empty:
            med = float(matches["selling_price"].median())
            q1 = float(matches["selling_price"].quantile(.25))
            q3 = float(matches["selling_price"].quantile(.75))
            pct_med = ((price-med)/med*100) if med else 0
            story.append(Paragraph(
                f"CarWise found <b>{len(matches):,}</b> historical records for the same brand and model "
                f"within an age window of ±2 years. Their median recorded price was <b>{money(med)}</b>, "
                f"with a middle-50% range of <b>{money(q1)}–{money(q3)}</b>. "
                f"The model estimate is {pct_med:+.1f}% relative to that historical median.",
                styles["CWBody"]
            ))
        else:
            story.append(Paragraph(
                "No sufficiently close same-brand, same-model historical group was found for this age window. "
                "The model estimate is available, but a direct comparable-car summary is not.",
                styles["CWBody"]
            ))

        story.append(PageBreak())
        story += [
            Paragraph("ANALYSIS & BUYER NOTES", styles["CWBrand"]),
            Paragraph("Key factors and practical next steps", styles["CWSub"]),
            Paragraph("What influences the model", styles["CWSection"])
        ]
        if feature_importance is not None and not feature_importance.empty:
            top = feature_importance.head(4).sort_values("importance_pct", ascending=True)
            drawing = Drawing(174*mm, 39*mm)
            chart = HorizontalBarChart()
            chart.x = 48
            chart.y = 4
            chart.width = 112*mm
            chart.height = 32*mm
            chart.data = [top["importance_pct"].astype(float).tolist()]
            chart.categoryAxis.categoryNames = top["feature"].astype(str).tolist()
            chart.categoryAxis.labels.fontSize = 7
            chart.valueAxis.labels.fontSize = 6
            chart.valueAxis.valueMin = 0
            chart.bars[0].fillColor = olive
            chart.barWidth = 5
            drawing.add(chart)
            story.append(drawing)
            story.append(Paragraph(
                "This is global Random Forest feature importance across the training data, not a per-car "
                "SHAP explanation. It shows which inputs were useful to the model overall; it does not prove causation.",
                styles["CWNote"]
            ))
        else:
            story.append(Paragraph(
                "Feature importance was not available from the local model artifact.",
                styles["CWNote"]
            ))

        story.append(Paragraph("Comparable-car snapshot", styles["CWSection"]))
        if matches is not None and not matches.empty:
            sample = matches[["car_age","km_driven","selling_price"]].copy()
            sample = sample.dropna()
            if not sample.empty:
                rows = [["Age (years)", "Kilometres", "Recorded price"]]
                for _, row in sample.sort_values("selling_price").iloc[
                    [0, len(sample)//2, len(sample)-1]
                ].iterrows():
                    rows.append([f"{row['car_age']:.0f}", f"{row['km_driven']:,.0f}", money(float(row["selling_price"]))])
                comp = Table(rows, colWidths=[45*mm, 60*mm, 71*mm])
                comp.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),ink), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                    ("GRID",(0,0),(-1,-1),0.35,line), ("FONTSIZE",(0,0),(-1,-1),7.5),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,paper]),
                    ("PADDING",(0,0),(-1,-1),5)
                ]))
                story.append(comp)
        else:
            story.append(Paragraph("Comparable historical records were not available.", styles["CWNote"]))

        story.append(Paragraph("Before you make a decision", styles["CWSection"]))
        tips = [
            "Verify RC, ownership, insurance and service records.",
            "Inspect accident damage, repainting, rust, tyres and suspension.",
            "Check engine, gearbox, clutch, brakes and air-conditioning.",
            "Take a test drive and arrange an independent mechanic inspection.",
            "Compare current local listings; historical dataset prices are not live Chennai offers."
        ]
        tip_rows = [[f"• {t}"] for t in tips]
        tip_table = Table(tip_rows, colWidths=[176*mm])
        tip_table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),paper), ("BOX",(0,0),(-1,-1),0.5,line),
            ("FONTSIZE",(0,0),(-1,-1),7.5), ("TEXTCOLOR",(0,0),(-1,-1),ink),
            ("LEFTPADDING",(0,0),(-1,-1),7), ("RIGHTPADDING",(0,0),(-1,-1),7),
            ("TOPPADDING",(0,0),(-1,-1),4), ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ]))
        story.append(tip_table)
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph(
            "Model: V3.1 Random Forest · Historical CarDekho dataset · Car age uses the project's fixed 2020 reference year. "
            "This report is decision support, not a certified appraisal. The model does not inspect condition, accident "
            "history, documents, location or live inventory. Do not interpret its estimate as a guaranteed sale price.",
            styles["CWNote"]
        ))
        doc.build(story)
        return buffer.getvalue()
    except Exception:
        return None


def save_prediction_history(car: dict[str, Any], result: dict[str, Any]) -> None:
    history = st.session_state.get("prediction_history", [])
    history.insert(0, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vehicle": f"{car['brand']} {car['car_model']} {car['variant_details']}".strip(),
        "predicted_price": float(result["predicted_price"]),
        "car": dict(car),
        "result": dict(result),
    })
    st.session_state["prediction_history"] = history[:30]


# ---------------------------------------------------------------------
# 4. SIDEBAR AND SHARED STATUS
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <div style="font-size: 1.4rem; font-weight: 800; color: #FFF; letter-spacing: -0.03em;">
                ⚡ CARWISE<span style="color: #FF5722;">.IN</span>
            </div>
            <div style="font-size: 0.65rem; font-weight: 700; color: #FF9800; letter-spacing: 0.15em; margin-top: 2px;">
                AUTOMOTIVE INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "AI Price Estimator",
            "Market Intelligence",
            "Compare Cars",
            "Chennai Car Hotspots",
            "Model Performance",
            "Dataset & Methodology",
            "Prediction History & Reports",
        ],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**Production model**")
    st.write("Version: `V3.1`")
    st.write("Algorithm: `Random Forest`")
    st.write("Training records: `6,926`")
    st.write("Input features: `13`")
    st.caption("Car age reference year: 2020")

    api_ok, api_message = check_api()
    if api_ok:
        st.success(api_message)
    else:
        st.warning(api_message)

    if st.button("Refresh app data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption("Historical dataset analysis · Machine-learning valuation")


# ---------------------------------------------------------------------
# 5. PAGE: OVERVIEW
# ---------------------------------------------------------------------

if page == "Overview":
    page_header(
        "CARWISE INDIA / MARKET BRIEF",
        "Understand the used-car market.",
        "A data-driven workspace for exploring historical used-car listings and estimating resale values with a trained machine-learning model.",
    )

    if df is None:
        show_empty_data_message()
    else:
        median_price = float(df["selling_price"].median())
        brands_count = int(df["brand"].nunique())
        models_count = int(df["car_model"].nunique())

        cols = st.columns(4)
        with cols[0]: kpi("Vehicles analysed", f"{len(df):,}", "Deduplicated V3.1 modelling dataset")
        with cols[1]: kpi("Median listing price", short_money(median_price), "Less sensitive to extreme prices")
        with cols[2]: kpi("Brands represented", f"{brands_count:,}", "Distinct brand labels")
        with cols[3]: kpi("Model families", f"{models_count:,}", "Extracted model categories")

        st.markdown('<div class="cw-section">Market snapshot</div>', unsafe_allow_html=True)
        left, right = st.columns([1.15, 1])
        with left:
            price_data = df.loc[df["selling_price"] > 0, ["selling_price"]].copy()
            fig = px.histogram(
                price_data, x="selling_price", nbins=45,
                title="Distribution of historical selling prices",
                labels={"selling_price": "Selling price (INR)", "count": "Number of listings"},
                marginal="box",
                color_discrete_sequence=[ACCENT_PRIMARY],
            )
            fig.update_xaxes(tickformat=",.0f")
            st.plotly_chart(plot_layout(fig, 420), use_container_width=True)
            show_graph_notes(fig)
        with right:
            brand_counts = df["brand"].value_counts().head(12).sort_values()
            fig = px.bar(
                x=brand_counts.values, y=brand_counts.index, orientation="h",
                title="Most represented brands",
                labels={"x": "Number of records", "y": "Brand"},
                color=brand_counts.values, color_continuous_scale=[[0, "#E9D8CC"], [1, "#B65E3C"]],
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(plot_layout(fig, 420), use_container_width=True)
            show_graph_notes(fig)
            
        st.markdown('<div class="cw-section">Price by market segment</div>', unsafe_allow_html=True)
        bins = [0, 500_000, 1_500_000, 3_000_000, np.inf]
        labels = ["Budget (< ₹5L)", "Mid-range (₹5–15L)", "Premium (₹15–30L)", "Luxury (₹30L+)"]
        segment = df.copy()
        segment["segment"] = pd.cut(segment["selling_price"], bins=bins, labels=labels, right=False)
        seg_stats = (
            segment.groupby("segment", observed=False)["selling_price"]
            .agg(records="count", median_price="median", mean_price="mean")
            .reset_index()
        )
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                seg_stats, x="segment", y="records", title="Records by price segment",
                labels={"segment": "Price segment", "records": "Records"},
                color="records", color_continuous_scale=[[0, "#E9E5D9"], [1, "#B65E3C"]],
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(plot_layout(fig, 350), use_container_width=True)
            show_graph_notes(fig)
        with c2:
            fig = px.bar(
                seg_stats, x="segment", y="median_price", title="Median price by segment",
                labels={"segment": "Price segment", "median_price": "Median price (INR)"},
                color="median_price", color_continuous_scale=[[0, "#E2E7DF"], [1, "#718477"]],
            )
            fig.update_yaxes(tickformat=",.0f")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(plot_layout(fig, 350), use_container_width=True)
            show_graph_notes(fig)
            
        st.markdown(
            '<div class="cw-note">Interpretation: these charts describe the historical dataset, '
            'not the complete current Indian used-car market. The dataset contains listing observations '
            'from an older period; do not read the median as a November 2026 market quote.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------
# 6. PAGE: AI PRICE ESTIMATOR
# ---------------------------------------------------------------------

elif page == "AI Price Estimator":
    page_header(
        "VALUATION / PREDICT",
        "Estimate a car's resale value.",
        "Enter the vehicle details below. The form uses categories observed in the V3.1 training dataset and sends the request to the FastAPI prediction service.",
    )

    if df is None:
        show_empty_data_message()
    else:
        st.markdown(
            '<div class="cw-note">Age convention: this model was trained with '
            '<b>car_age = 2020 − manufacturing year</b>. For example, a 2015 car should be entered as age 5. '
            'Use the same convention for consistent predictions.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Vehicle profile")
        # Keep dependent dropdowns outside st.form so each selection reruns the
        # app immediately (Brand -> Model -> Variant).
        car = render_car_form("estimate_v4", df)
        asking_price_input = st.number_input(
            "Seller's asking price (optional, ₹)",
            min_value=0, max_value=100_000_000, value=0, step=10_000,
            key="estimate_v4_asking_price",
            help="Enter the seller's quote to compare it with the model estimate. Leave 0 if unknown."
        )
        submitted = st.button(
            "Generate valuation", type="primary", use_container_width=True,
            key="generate_car_valuation_v4",
        )

        if submitted:
            with st.spinner("Sending vehicle details to the valuation model..."):
                result = predict_car(car)
            if result:
                asking_price = float(asking_price_input) if asking_price_input > 0 else None
                st.session_state["last_prediction"] = {
                    "car": car, "result": result, "asking_price": asking_price
                }
                save_prediction_history(car, result)
                st.success("Valuation generated.")
        saved = st.session_state.get("last_prediction")
        if saved:
            st.markdown("### Valuation result")
            show_prediction(saved["result"], saved["car"])
            asking_price = saved.get("asking_price")
            if asking_price:
                estimate = float(saved["result"]["predicted_price"])
                gap_pct = (asking_price - estimate) / estimate * 100 if estimate else 0
                verdict = "Above model estimate" if gap_pct > 5 else ("Below model estimate" if gap_pct < -5 else "Close to model estimate")
                d1, d2, d3 = st.columns(3)
                d1.metric("Seller asking price", money(asking_price))
                d2.metric("Difference from estimate", money(abs(asking_price-estimate)), f"{gap_pct:+.1f}%")
                d3.metric("Price comparison", verdict)
                st.caption("This is a comparison with the model estimate, not a verified market-fairness judgement.")
            st.markdown("### Why does CarWise estimate this price?")
            st.write(
                "The valuation is produced from patterns learned from historical vehicle records. "
                "The charts below put this estimate in context; they are descriptive comparisons, "
                "not proof that one feature causes a price change."
            )
            saved_car = saved["car"]
            saved_price = float(saved["result"]["predicted_price"])
            matches = historical_comparison(df, saved_car, saved_price)
            if not matches.empty:
                median_price = float(matches["selling_price"].median())
                low_price = float(matches["selling_price"].quantile(0.25))
                high_price = float(matches["selling_price"].quantile(0.75))
                delta_pct = ((saved_price - median_price) / median_price * 100) if median_price else 0.0
                verdict = "Above historical middle range" if saved_price > high_price else ("Below historical middle range" if saved_price < low_price else "Within historical middle range")
                m1, m2, m3 = st.columns(3)
                m1.metric("Similar-car historical median", money(median_price))
                m2.metric("Middle 50% range", f"{short_money(low_price)}–{short_money(high_price)}")
                m3.metric("Estimate vs median", f"{delta_pct:+.1f}%")
                st.info(f"**Context, not a live-market verdict:** {verdict}. Based on {len(matches):,} records with the same brand and model family and an age window of ±2 years. These are historical dataset records, not current Chennai listings.")

                left, right = st.columns(2)
                with left:
                    age_fig = px.scatter(df, x="car_age", y="selling_price", color="fuel" if "fuel" in df.columns else None,
                                         opacity=0.42, title="How vehicle age relates to recorded price",
                                         labels={"car_age":"Car age (2020 reference)","selling_price":"Recorded selling price (₹)"})
                    age_fig.add_vline(x=float(saved_car["car_age"]), line_dash="dash", line_color=ACCENT_PRIMARY,
                                      annotation_text="Your car age")
                    age_fig.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(plot_layout(age_fig, 350), use_container_width=True)
                    st.markdown("**What to notice**")
                    st.markdown("- Compare the selected car's age with the wider dataset.\n- Older vehicles often occupy lower price bands, although model and specifications matter.\n- The vertical line marks the age entered for this valuation.")
                with right:
                    km_fig = px.scatter(df, x="km_driven", y="selling_price", color="fuel" if "fuel" in df.columns else None,
                                         opacity=0.42, title="How kilometres driven relate to recorded price",
                                         labels={"km_driven":"Kilometres driven","selling_price":"Recorded selling price (₹)"})
                    km_fig.add_vline(x=float(saved_car["km_driven"]), line_dash="dash", line_color=ACCENT_PRIMARY,
                                     annotation_text="Your car")
                    km_fig.update_xaxes(tickformat=",.0f")
                    km_fig.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(plot_layout(km_fig, 350), use_container_width=True)
                    st.markdown("**What to notice**")
                    st.markdown("- Higher usage can be associated with lower resale prices.\n- Condition and maintenance are not captured by kilometres alone.\n- The vertical line marks the entered odometer reading.")

                hist = px.histogram(matches, x="selling_price", nbins=22,
                                    title="Historical price distribution for similar cars",
                                    labels={"selling_price":"Recorded selling price (₹)"},
                                    color_discrete_sequence=[ACCENT_SECONDARY])
                hist.add_vline(x=saved_price, line_dash="dash", line_color=ACCENT_PRIMARY,
                               annotation_text="CarWise estimate")
                hist.add_vline(x=median_price, line_dash="dot", line_color="#536B78",
                               annotation_text="Historical median")
                hist.update_xaxes(tickformat=",.0f")
                st.plotly_chart(plot_layout(hist, 370), use_container_width=True)
                st.markdown("**How to read this:** the bars show historical recorded prices for same-brand, same-model cars within ±2 years of the entered age. The dashed line is the model estimate; the dotted line is the historical median.")
            else:
                st.info("There are no same-brand, same-model records within the selected ±2-year age window. The estimate is still available, but a similar-car distribution cannot be shown reliably.")

            st.markdown("### What the trained model pays attention to")
            importance = get_feature_importance()
            if importance is not None and not importance.empty:
                top = importance.head(7).sort_values("importance_pct", ascending=True)
                imp_fig = px.bar(top, x="importance_pct", y="feature", orientation="h",
                                 title="Global feature importance — across the training dataset",
                                 labels={"importance_pct":"Share of total importance (%)","feature":"Input feature"},
                                 color_discrete_sequence=[ACCENT_PRIMARY])
                st.plotly_chart(plot_layout(imp_fig, 340), use_container_width=True)
                st.markdown("- This graph describes the model overall, not the exact reason for this one prediction.\n- Importance indicates how useful a feature was to the Random Forest splits.\n- It does not establish causation or guarantee that changing a feature changes price by a fixed amount.")
            else:
                st.caption("Global feature importance is unavailable because the trained model artifact could not be loaded.")

            with st.expander("Similar historical cars — inspect the records"):
                if matches.empty:
                    st.info("No comparable historical records were found.")
                else:
                    st.dataframe(
                        matches[["brand", "car_model", "variant_details", "car_age", "km_driven", "selling_price"]]
                        .sort_values("selling_price", ascending=False).head(20),
                        use_container_width=True, hide_index=True,
                    )
            with st.expander("Before you buy: practical inspection checklist"):
                checklist = ["Verify RC, chassis and engine numbers", "Check service history and ownership records",
                             "Inspect for accident damage, repainting and rust", "Check engine, gearbox, clutch and suspension",
                             "Inspect tyres, brakes, lights and air-conditioning", "Verify insurance, PUC and pending challans",
                             "Take a test drive and arrange an independent mechanic inspection"]
                for item in checklist:
                    st.checkbox(item, key="buyer_check_" + str(abs(hash(item))))

            pdf_bytes = build_valuation_pdf(
                saved["car"], saved["result"], saved.get("asking_price"),
                matches=matches, feature_importance=importance
            )
            if pdf_bytes:
                st.download_button(
                    "Download valuation report (PDF)",
                    data=pdf_bytes,
                    file_name="carwise_valuation_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.info("PDF export needs ReportLab. Install it with: python -m pip install reportlab")


# ---------------------------------------------------------------------
# 7. PAGE: MARKET INTELLIGENCE
# ---------------------------------------------------------------------

elif page == "Market Intelligence":
    page_header(
        "MARKET / EXPLORE",
        "Explore patterns behind car prices.",
        "Filter the historical dataset and inspect how price distributions vary across brands, models, specifications and vehicle characteristics.",
    )

    if df is None:
        show_empty_data_message()
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            selected_brands = st.multiselect(
                "Brands", sorted(df["brand"].dropna().astype(str).unique()),
                default=[],
            )
        with f2:
            fuel_values = sorted(df["fuel"].dropna().astype(str).unique())
            selected_fuels = st.multiselect("Fuel types", fuel_values, default=[])
        with f3:
            transmission_values = sorted(df["transmission"].dropna().astype(str).unique())
            selected_transmissions = st.multiselect("Transmission", transmission_values, default=[])

        filtered = df.copy()
        if selected_brands:
            filtered = filtered[filtered["brand"].astype(str).isin(selected_brands)]
        if selected_fuels:
            filtered = filtered[filtered["fuel"].astype(str).isin(selected_fuels)]
        if selected_transmissions:
            filtered = filtered[filtered["transmission"].astype(str).isin(selected_transmissions)]

        if filtered.empty:
            st.warning("No records match the selected filters.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Filtered records", f"{len(filtered):,}")
            c2.metric("Median price", money(float(filtered["selling_price"].median())))
            c3.metric("Mean price", money(float(filtered["selling_price"].mean())))
            c4.metric("Brands", f"{filtered['brand'].nunique():,}")

            tab1, tab2, tab3, tab4 = st.tabs(
                ["Price landscape", "Brand & model", "Vehicle profile", "Relationships"]
            )
            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(
                        filtered, x="selling_price", nbins=40, marginal="box",
                        title="Selling price distribution",
                        labels={"selling_price": "Selling price (INR)"},
                        color_discrete_sequence=[ACCENT_PRIMARY],
                    )
                    fig.update_xaxes(tickformat=",.0f")
                    st.plotly_chart(plot_layout(fig), use_container_width=True)
                    show_graph_notes(fig)
                with c2:
                    fig = px.box(
                        filtered, x="fuel", y="selling_price", color="fuel",
                        points="outliers", title="Price distribution by fuel type",
                        labels={"fuel": "Fuel", "selling_price": "Selling price (INR)"},
                    )
                    fig.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(plot_layout(fig), use_container_width=True)
                    show_graph_notes(fig)

                seg = filtered.copy()
                seg["price_segment"] = pd.cut(
                    seg["selling_price"],
                    bins=[0, 500_000, 1_500_000, 3_000_000, np.inf],
                    labels=["Budget", "Mid-range", "Premium", "Luxury"],
                    right=False,
                )
                counts = seg["price_segment"].value_counts().reindex(
                    ["Budget", "Mid-range", "Premium", "Luxury"], fill_value=0
                ).rename_axis("Segment").reset_index(name="Records")
                fig = px.pie(
                    counts, names="Segment", values="Records", hole=.55,
                    title="Share of filtered records by price segment",
                )
                st.plotly_chart(plot_layout(fig, 350), use_container_width=True)
                show_graph_notes(fig)
            with tab2:
                brand_stats = (
                    filtered.groupby("brand")["selling_price"]
                    .agg(records="count", median_price="median", mean_price="mean")
                    .query("records >= 5")
                    .sort_values("median_price", ascending=False)
                    .reset_index()
                )
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(
                        brand_stats.head(18).sort_values("median_price"),
                        x="median_price", y="brand", orientation="h",
                        color="median_price", color_continuous_scale=[[0, "#E9D8CC"], [1, "#B65E3C"]],
                        title="Brands ranked by median price (minimum 5 records)",
                        labels={"median_price": "Median price (INR)", "brand": "Brand"},
                    )
                    fig.update_xaxes(tickformat=",.0f")
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(plot_layout(fig, 520), use_container_width=True)
                    show_graph_notes(fig)
                with c2:
                    fig = px.scatter(
                        brand_stats, x="records", y="median_price", size="records",
                        hover_name="brand", title="Brand volume vs median price",
                        labels={"records": "Number of records", "median_price": "Median price (INR)"},
                        color="median_price", color_continuous_scale=[[0, "#E2E7DF"], [1, "#718477"]],
                    )
                    fig.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(plot_layout(fig, 520), use_container_width=True)
                    show_graph_notes(fig)
                model_stats = (
                    filtered.groupby(["brand", "car_model"])["selling_price"]
                    .agg(records="count", median_price="median", mean_price="mean")
                    .query("records >= 5")
                    .sort_values("median_price", ascending=False)
                    .reset_index()
                )
                st.markdown("**Model families with at least five records**")
                st.dataframe(model_stats.head(30), use_container_width=True, hide_index=True)

            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.box(
                        filtered, x="transmission", y="selling_price",
                        color="transmission", points="outliers",
                        title="Manual vs automatic price distribution",
                        labels={"selling_price": "Selling price (INR)", "transmission": "Transmission"},
                    )
                    fig.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(plot_layout(fig), use_container_width=True)
                    show_graph_notes(fig)
                with c2:
                    age_stats = (
                        filtered.groupby("car_age")["selling_price"]
                        .agg(median_price="median", q1=lambda x: x.quantile(.25),
                             q3=lambda x: x.quantile(.75), records="count")
                        .reset_index()
                    )
                    age_stats = age_stats[age_stats["records"] >= 3]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=age_stats["car_age"], y=age_stats["q3"],
                        line=dict(width=0), showlegend=False, name="Q3",
                    ))
                    fig.add_trace(go.Scatter(
                        x=age_stats["car_age"], y=age_stats["q1"],
                        fill="tonexty", fillcolor="rgba(113, 132, 119, 0.16)",
                        line=dict(width=0), name="Middle 50% range",
                    ))
                    fig.add_trace(go.Scatter(
                        x=age_stats["car_age"], y=age_stats["median_price"],
                        mode="lines+markers", line=dict(color=ACCENT_PRIMARY, width=2.5),
                        name="Median price",
                    ))
                    fig.update_layout(title="Age vs price: median and interquartile range",
                                      xaxis_title="Car age (2020 reference)", yaxis_title="Selling price (INR)")
                    fig.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(plot_layout(fig), use_container_width=True)
                    show_graph_notes(fig)

                fuel_age = (
                    filtered.groupby(["car_age", "fuel"], observed=True)["selling_price"]
                    .median().reset_index()
                )
                fig = px.line(
                    fuel_age, x="car_age", y="selling_price", color="fuel",
                    markers=True, title="Median price by age and fuel type",
                    labels={"car_age": "Car age (2020 reference)", "selling_price": "Median price (INR)"},
                )
                fig.update_yaxes(tickformat=",.0f")
                st.plotly_chart(plot_layout(fig), use_container_width=True)
                show_graph_notes(fig)

            with tab4:
                scatter = filtered[["km_driven", "selling_price", "max_power", "engine", "mileage", "fuel"]].dropna()
                if len(scatter) > 2500:
                    scatter = scatter.sample(2500, random_state=42)
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.scatter(
                        scatter, x="km_driven", y="selling_price",
                        color="max_power", opacity=.62,
                        title="Kilometres driven vs selling price",
                        labels={"km_driven": "Kilometres driven", "selling_price": "Selling price (INR)",
                                "max_power": "Max power (bhp)"},
                        color_continuous_scale=[[0, "#D9DED7"], [1, "#718477"]],
                    )
                    fig.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(plot_layout(fig), use_container_width=True)
                    show_graph_notes(fig)
                with c2:
                    fig = px.scatter(
                        scatter, x="max_power", y="selling_price",
                        color="engine", opacity=.62,
                        title="Maximum power vs selling price",
                        labels={"max_power": "Max power (bhp)", "selling_price": "Selling price (INR)",
                                "engine": "Engine (cc)"},
                        color_continuous_scale=[[0, "#E5E2D8"], [1, "#536B78"]],
                    )
                    fig.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(plot_layout(fig), use_container_width=True)
                    show_graph_notes(fig)

                c1, c2 = st.columns(2)
                with c1:
                    fig = px.scatter(
                        scatter, x="mileage", y="selling_price",
                        color="fuel", opacity=.65,
                        title="Mileage vs selling price",
                        labels={"mileage": "Mileage (km/l)", "selling_price": "Selling price (INR)"},
                    )
                    fig.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(plot_layout(fig), use_container_width=True)
                    show_graph_notes(fig)
                with c2:
                    corr_cols = ["car_age", "km_driven", "mileage", "engine", "max_power", "seats", "selling_price"]
                    corr = filtered[corr_cols].corr(method="spearman", numeric_only=True)
                    fig = px.imshow(
                        corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Spearman rank-correlation heatmap",
                    )
                    st.plotly_chart(plot_layout(fig, 440), use_container_width=True)
                    show_graph_notes(fig)
                st.caption(
                    "These are observational associations in historical listings. "
                    "Correlation does not establish that a specification causes a price change."
                )


# ---------------------------------------------------------------------
# 8. PAGE: COMPARE CARS
# ---------------------------------------------------------------------

elif page == "Compare Cars":
    page_header(
        "COMPARISON / DECISION SUPPORT",
        "Compare two vehicles side by side.",
        "Generate two model estimates and compare the specifications. A higher predicted resale value does not automatically make a car the better purchase.",
    )

    if df is None:
        show_empty_data_message()
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 🚘 Vehicle A")
            car_a = render_car_form("compareA_v4", df)
        with col_b:
            st.markdown("### 🚙 Vehicle B")
            car_b = render_car_form("compareB_v4", df)
        compare = st.button(
            "Compare vehicle estimates", type="primary", use_container_width=True,
            key="compare_vehicle_estimates_v4",
        )

        if compare:
            with st.spinner("Requesting both valuations..."):
                result_a = predict_car(car_a)
                result_b = predict_car(car_b)
            if result_a and result_b:
                price_a = result_a["predicted_price"]
                price_b = result_b["predicted_price"]
                st.session_state["comparison_result"] = {
                    "car_a": car_a, "car_b": car_b,
                    "price_a": price_a, "price_b": price_b,
                }

        result = st.session_state.get("comparison_result")
        if result:
            a, b = result["car_a"], result["car_b"]
            pa, pb = result["price_a"], result["price_b"]
            st.markdown("### Estimated resale values")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f'<div class="cw-result"><div class="cw-result-label">Vehicle A · {a["brand"]} {a["car_model"]}</div>'
                    f'<div class="cw-result-price">{money(pa)}</div><div style="color:#94A3B8">{a["variant_details"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="cw-result"><div class="cw-result-label">Vehicle B · {b["brand"]} {b["car_model"]}</div>'
                    f'<div class="cw-result-price">{money(pb)}</div><div style="color:#94A3B8">{b["variant_details"]}</div></div>',
                    unsafe_allow_html=True,
                )
            diff = abs(pa - pb)
            higher = "Vehicle A" if pa > pb else "Vehicle B" if pb > pa else "Neither"
            c1, c2, c3 = st.columns(3)
            c1.metric("Absolute estimate difference", money(diff))
            c2.metric("Higher estimated value", higher)
            c3.metric("Difference relative to lower", f"{diff / min(pa, pb) * 100:.1f}%" if min(pa, pb) else "—")

            comparison_df = pd.DataFrame({
                "Specification": [
                    "Brand", "Model", "Variant", "Car age (2020 basis)",
                    "Kilometres driven", "Mileage (km/l)", "Engine (cc)",
                    "Max power (bhp)", "Seats", "Fuel", "Transmission", "Owner",
                    "Predicted price",
                ],
                "Vehicle A": [
                    a["brand"], a["car_model"], a["variant_details"], a["car_age"],
                    a["km_driven"], a["mileage"], a["engine"], a["max_power"],
                    a["seats"], a["fuel"], a["transmission"], a["owner"], money(pa),
                ],
                "Vehicle B": [
                    b["brand"], b["car_model"], b["variant_details"], b["car_age"],
                    b["km_driven"], b["mileage"], b["engine"], b["max_power"],
                    b["seats"], b["fuel"], b["transmission"], b["owner"], money(pb),
                ],
            })
            st.markdown("### Specification comparison")
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            fig = go.Figure(data=[
                go.Bar(name="Estimated value", x=["Vehicle A", "Vehicle B"], y=[pa, pb],
                       marker_color=[ACCENT_PRIMARY, ACCENT_CYAN], text=[money(pa), money(pb)], textposition="outside")
            ])
            fig.update_layout(title="Predicted resale value comparison", yaxis_title="Estimated price (INR)")
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(plot_layout(fig, 350), use_container_width=True)
            show_graph_notes(fig)
            st.warning(
                "This comparison ranks model estimates only. It does not assess mechanical condition, "
                "accident history, service records, legal status, insurance, or total ownership cost."
            )


# ---------------------------------------------------------------------
# 9. PAGE: CHENNAI USED-CAR HOTSPOTS
# ---------------------------------------------------------------------

elif page == "Chennai Car Hotspots":
    page_header(
        "LOCAL DISCOVERY / CHENNAI",
        "Explore used-car shopping areas.",
        "Explore candidate dealership clusters around Chennai. The circles show approximate area search zones, not verified dealer premises or price guarantees."
    )
    st.markdown(
        '<div class="cw-note"><b>How to use this map:</b> each circle is an approximate neighbourhood search zone. '
        'Use the Google Maps search link to find current dealerships, verify their details and compare actual listings before visiting.</div>',
        unsafe_allow_html=True,
    )
    hotspots = [
        {"name":"Guindy", "lat":13.0067, "lon":80.2206, "radius":1300, "note":"Central Chennai access; search for pre-owned dealerships and multi-brand showrooms."},
        {"name":"Porur", "lat":13.0358, "lon":80.1560, "radius":1500, "note":"West Chennai candidate area; compare dealer inventory and vehicle history."},
        {"name":"Velachery", "lat":12.9750, "lon":80.2210, "radius":1300, "note":"South Chennai candidate area; check nearby listings and service support."},
        {"name":"Anna Nagar", "lat":13.0850, "lon":80.2101, "radius":1300, "note":"North-west/central residential catchment; verify dealer locations before travel."},
        {"name":"Tambaram", "lat":12.9249, "lon":80.1000, "radius":1600, "note":"South-west corridor candidate area; compare multiple sellers and total ownership costs."},
        {"name":"Ambattur", "lat":13.1143, "lon":80.1548, "radius":1500, "note":"North-west Chennai candidate area; use the map search to identify current showrooms."},
        {"name":"OMR / Perungudi", "lat":12.9592, "lon":80.2425, "radius":1500, "note":"IT corridor candidate area; check current inventory and seller documentation."},
        {"name":"Pallikaranai", "lat":12.9380, "lon":80.2120, "radius":1300, "note":"South Chennai candidate area; verify availability and compare asking prices."},
    ]
    selected_area = st.selectbox("Focus area", ["All Chennai areas"] + [x["name"] for x in hotspots], key="chennai_hotspot_area")
    visible = hotspots if selected_area == "All Chennai areas" else [x for x in hotspots if x["name"] == selected_area]
    try:
        import folium
        from streamlit_folium import st_folium
        center = [13.045, 80.19]
        fmap = folium.Map(location=center, zoom_start=11 if len(visible) > 1 else 13,
                          tiles="OpenStreetMap", control_scale=True)
        for item in visible:
            folium.Circle(
                location=[item["lat"], item["lon"]], radius=item["radius"],
                color="#B65E3C", weight=2, fill=True, fill_color="#D9A58E",
                fill_opacity=0.20, tooltip=f"{item['name']} · approximate search zone",
            ).add_to(fmap)
            folium.Marker(
                [item["lat"], item["lon"]],
                tooltip=item["name"],
                popup=folium.Popup(f"<b>{item['name']}</b><br>{item['note']}<br><a href='https://www.google.com/maps/search/used+car+dealers+{item['name'].replace(' ', '+')}+Chennai' target='_blank'>Search dealers on Google Maps</a>", max_width=300),
                icon=folium.Icon(color="darkred", icon="car", prefix="fa"),
            ).add_to(fmap)
        st_folium(fmap, use_container_width=True, height=560, returned_objects=[])
    except ImportError:
        st.warning("Install the map packages to display the hotspot circles: python -m pip install folium streamlit-folium")
        st.info("Google Maps search links are still available below.")

    st.markdown("### Explore areas and find current dealerships")
    cards = st.columns(2)
    for i, item in enumerate(visible):
        with cards[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{item['name']}**")
                st.caption(item["note"])
                gmaps_url = f"https://www.google.com/maps/search/?api=1&query=used+car+dealers+{item['name'].replace(' ', '+')}+Chennai"
                st.link_button("Find used-car dealers on Google Maps", gmaps_url, use_container_width=True)
    st.caption("The highlighted circles are approximate neighbourhood search zones. They do not represent surveyed dealer density, verified cheapest-price zones, or endorsements. Dealer ratings, stock and prices must be checked on the linked live listings.")


# ---------------------------------------------------------------------
# 10. PAGE: MODEL PERFORMANCE
# ---------------------------------------------------------------------

elif page == "Model Performance":
    page_header(
        "MACHINE LEARNING / VALIDATION",
        "Inspect model performance.",
        "Review the algorithm experiment, holdout metrics, cross-validation results and global feature importance. Lower MAE/RMSE and higher R² are generally preferred.",
    )

    test_results = load_results(str(TEST_RESULTS_PATH), file_stamp(TEST_RESULTS_PATH)) if TEST_RESULTS_PATH.exists() else None
    cv_results = load_results(str(CV_RESULTS_PATH), file_stamp(CV_RESULTS_PATH)) if CV_RESULTS_PATH.exists() else None

    if test_results is None:
        st.error("Algorithm test-results CSV is missing. Expected artifacts/v3_1_algorithm_test_results.csv.")
    else:
        test_results.columns = [str(c).strip() for c in test_results.columns]
        needed = {"Algorithm", "MAE", "RMSE", "R2"}
        if not needed.issubset(test_results.columns):
            st.error(f"Test-results file must contain columns: {sorted(needed)}")
            st.dataframe(test_results, use_container_width=True)
        else:
            for col in ["MAE", "RMSE", "R2"]:
                test_results[col] = pd.to_numeric(test_results[col], errors="coerce")
            test_results = test_results.dropna(subset=["MAE", "RMSE", "R2"])
            rf = test_results[test_results["Algorithm"].astype(str).str.contains("Random Forest", case=False)]
            st.markdown("### Production model · Random Forest")
            if not rf.empty:
                row = rf.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Holdout MAE", money(float(row["MAE"])))
                c2.metric("Holdout RMSE", money(float(row["RMSE"])))
                c3.metric("Holdout R²", f"{float(row['R2']):.4f}")
                c4.metric("Holdout records", "1,385")
            else:
                st.warning("Random Forest row was not found in the test-results file.")

            st.caption(
                "The holdout metrics shown are from the algorithm comparison experiment. "
                "The final production pipeline was subsequently refit on all 6,926 records; "
                "these holdout metrics are not a fresh evaluation of that all-data fitted object."
            )

            st.markdown("### Algorithm comparison on the experiment holdout")
            display = test_results.sort_values("MAE").copy()
            st.dataframe(display, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    display, x="Algorithm", y="MAE", color="Algorithm",
                    title="Holdout MAE (lower is better)",
                    labels={"MAE": "MAE (INR)"},
                )
                fig.update_yaxes(tickformat=",.0f")
                st.plotly_chart(plot_layout(fig, 380), use_container_width=True)
                show_graph_notes(fig)
            with c2:
                fig = px.bar(
                    display, x="Algorithm", y="RMSE", color="Algorithm",
                    title="Holdout RMSE (lower is better)",
                    labels={"RMSE": "RMSE (INR)"},
                )
                fig.update_yaxes(tickformat=",.0f")
                st.plotly_chart(plot_layout(fig, 380), use_container_width=True)
                show_graph_notes(fig)
            fig = px.bar(
                display, x="Algorithm", y="R2", color="Algorithm",
                title="Holdout R² (higher is better)",
                labels={"R2": "R²"},
            )
            st.plotly_chart(plot_layout(fig, 330), use_container_width=True)
            show_graph_notes(fig)
    st.markdown("### Grouped cross-validation")
    if cv_results is not None:
        st.dataframe(cv_results, use_container_width=True, hide_index=True)
        cv = cv_results.copy()
        cv.columns = [str(c).strip() for c in cv.columns]
        cv_mae_col = next((c for c in ["Best_CV_MAE", "CV_MAE", "Mean_CV_MAE"] if c in cv.columns), None)
        if "Algorithm" in cv.columns and cv_mae_col:
            cv[cv_mae_col] = pd.to_numeric(cv[cv_mae_col], errors="coerce")
            cv = cv.dropna(subset=[cv_mae_col]).sort_values(cv_mae_col)
            fig = px.bar(
                cv, x="Algorithm", y=cv_mae_col, color="Algorithm",
                title="Grouped cross-validation MAE (lower is better)",
                labels={cv_mae_col: "Cross-validation MAE (INR)"},
            )
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(plot_layout(fig, 350), use_container_width=True)
            show_graph_notes(fig)
        else:
            st.info("The CV CSV is available, but it does not contain a recognised summary MAE column.")
    else:
        st.info("Cross-validation summary file is not available.")

    st.markdown("### What the model learned")
    importance = get_feature_importance()
    if importance is None:
        st.info("Feature importance is unavailable. Check the production model file and pipeline structure.")
    else:
        fig = px.bar(
            importance.sort_values("importance_pct"),
            x="importance_pct", y="feature", orientation="h",
            color="importance_pct", color_continuous_scale=[[0, "#E9D8CC"], [1, "#B65E3C"]],
            title="Aggregated Random Forest feature importance",
            labels={"importance_pct": "Importance (%)", "feature": "Input feature"},
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(plot_layout(fig, 440), use_container_width=True)
        show_graph_notes(fig)
        st.dataframe(
            importance[["feature", "importance_pct"]].rename(
                columns={"feature": "Feature", "importance_pct": "Importance (%)"}
            ).round({"Importance (%)": 2}),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Random Forest impurity-based feature importance is a global model diagnostic. "
            "It is not a causal effect and does not explain an individual prediction."
        )

    with st.expander("Metric guide"):
        st.markdown(
            "- **MAE:** average absolute prediction error in rupees; lower is better.\n"
            "- **RMSE:** gives larger errors more weight; lower is better.\n"
            "- **R²:** measures fit relative to a mean-price baseline; closer to 1 is generally better.\n"
            "- **Grouped cross-validation:** evaluates performance across grouped folds, reducing the risk "
            "of closely related records appearing in both train and validation folds."
        )

    with st.expander("Important experiment limitation"):
        st.write(
            "The algorithm comparison used the holdout results to compare candidate algorithms. "
            "Because the holdout was then used to choose between candidates, it should not be described "
            "as a completely untouched final test after model selection. A stricter future evaluation "
            "would use nested cross-validation or a separate final holdout reserved until every model "
            "and hyperparameter decision is complete."
        )


# ---------------------------------------------------------------------
# 10. PAGE: DATASET AND METHODOLOGY
# ---------------------------------------------------------------------

elif page == "Dataset & Methodology":
    page_header(
        "TRANSPARENCY / DATA",
        "Know what the model is built on.",
        "A concise audit of the modelling dataset, feature set, assumptions and known limitations.",
    )

    if df is None:
        show_empty_data_message()
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns):,}")
        c3.metric("Brands", f"{df['brand'].nunique():,}")
        c4.metric("Model families", f"{df['car_model'].nunique():,}")

        st.markdown("### Dataset preview")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

        st.markdown("### Missing-value profile")
        missing = df.isna().sum().rename("missing_count").reset_index()
        missing.columns = ["feature", "missing_count"]
        missing["missing_pct"] = missing["missing_count"] / len(df) * 100
        missing = missing.sort_values("missing_pct", ascending=False)
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.bar(
                missing, x="missing_pct", y="feature", orientation="h",
                title="Missing values by feature",
                labels={"missing_pct": "Missing values (%)", "feature": "Feature"},
                color="missing_pct", color_continuous_scale=[[0, "#E9E5D9"], [1, "#B65E3C"]],
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(plot_layout(fig, 430), use_container_width=True)
            show_graph_notes(fig)
        with c2:
            st.dataframe(missing.round({"missing_pct": 2}), use_container_width=True, hide_index=True)

        st.markdown("### Feature groups")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Numerical inputs**")
            st.write(", ".join(NUMERIC_FEATURES))
        with c2:
            st.markdown("**Categorical inputs**")
            st.write(", ".join(CATEGORICAL_FEATURES))
        st.write("**Prediction target:** `selling_price` (INR)")

        st.markdown("### Methodology")
        st.markdown(
            """
            1. **Data preparation:** numeric vehicle specifications and categorical vehicle details
               are prepared for modelling.
            2. **Feature engineering:** the V3.1 feature set includes brand, model family and
               variant details alongside vehicle specifications.
            3. **Preprocessing:** numerical missing values are imputed; categorical values are
               imputed and one-hot encoded in the model pipeline.
            4. **Algorithm study:** Random Forest, Extra Trees and HistGradientBoosting were compared.
            5. **Production model:** the selected Random Forest pipeline was refit using the full
               6,926-record V3.1 modelling dataset.
            """
        )

        st.markdown("### Known limitations")
        st.warning(
            "- The dataset is historical and does not represent live November 2026 listings.\n"
            "- The car-age feature uses a fixed 2020 reference year; this convention is retained for model compatibility.\n"
            "- The model cannot inspect condition, accident history, maintenance, location or ownership documents.\n"
            "- Rare premium and luxury examples are limited, so predictions in these segments may be less reliable.\n"
            "- Model-family and variant labels are derived from vehicle names and may contain extraction simplifications.\n"
            "- The estimate is decision support, not a professional appraisal or guaranteed transaction price."
        )

        st.markdown("### Download filtered modelling data")
        st.download_button(
            "Download dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="carwise_v3_1_model_data.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------
# 11. PAGE: PREDICTION HISTORY AND REPORTS
# ---------------------------------------------------------------------

elif page == "Prediction History & Reports":
    page_header(
        "WORKSPACE / REPORTS",
        "Your valuation history.",
        "Review estimates generated during this browser session and download individual valuation reports.",
    )
    history = st.session_state.get("prediction_history", [])
    if not history:
        st.info("No predictions have been generated in this session yet. Open AI Price Estimator to create your first valuation.")
    else:
        total = len(history)
        avg_price = float(np.mean([item["predicted_price"] for item in history]))
        c1, c2, c3 = st.columns(3)
        c1.metric("Saved estimates", total)
        c2.metric("Average estimated value", money(avg_price))
        c3.metric("Latest estimate", money(history[0]["predicted_price"]))
        st.markdown("### Recent valuations")
        display = pd.DataFrame([
            {"Generated at": h["timestamp"], "Vehicle": h["vehicle"],
             "Estimated price": money(h["predicted_price"])}
            for h in history
        ])
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.markdown("### Export a report")
        selected = st.selectbox(
            "Choose a valuation",
            options=list(range(total)),
            format_func=lambda i: f"{history[i]['vehicle']} · {money(history[i]['predicted_price'])} · {history[i]['timestamp']}",
        )
        chosen = history[selected]
        pdf_bytes = build_valuation_pdf(chosen["car"], chosen["result"])
        if pdf_bytes:
            st.download_button(
                "Download selected valuation PDF",
                data=pdf_bytes,
                file_name="carwise_valuation_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning("PDF generation is unavailable. Install ReportLab: python -m pip install reportlab")
        if st.button("Clear session history", type="secondary"):
            st.session_state["prediction_history"] = []
            st.rerun()
        st.caption("History is stored only in the current Streamlit session. It is not saved to a database.")


# ---------------------------------------------------------------------
# 12. FOOTER
# ---------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#64748B;padding:16px 0 8px 0;font-size:0.85rem;">
      <b style="color:#FF5722;letter-spacing:0.05em;">CARWISE INDIA</b> · AUTOMOTIVE INTELLIGENCE ENGINE · V3.1 RANDOM FOREST
      <br><span style="font-size:0.78rem;color:#475569;">Historical-data-based used-car price estimation · Academic project</span>
    </div>
    """,
    unsafe_allow_html=True,
)