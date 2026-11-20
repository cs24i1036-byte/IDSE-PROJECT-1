"""Verify required CarWise libraries."""
import sys

def main():
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        import seaborn as sns
        import sklearn
        import joblib
        import fastapi
        import streamlit
        import requests
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Run: pip install -r requirements.txt")
        raise SystemExit(1) from exc

    print("=" * 48)
    print("CARWISE INDIA — ENVIRONMENT CHECK")
    print("=" * 48)
    print("Python       :", sys.version.split()[0])
    print("NumPy        :", np.__version__)
    print("Pandas       :", pd.__version__)
    print("Matplotlib   :", matplotlib.__version__)
    print("Seaborn      :", sns.__version__)
    print("Scikit-learn :", sklearn.__version__)
    print("Joblib       :", joblib.__version__)
    print("FastAPI      :", fastapi.__version__)
    print("Streamlit    :", streamlit.__version__)
    print("Requests     :", requests.__version__)
    print("=" * 48)
    print("All required libraries imported successfully!")

if __name__ == "__main__":
    main()
