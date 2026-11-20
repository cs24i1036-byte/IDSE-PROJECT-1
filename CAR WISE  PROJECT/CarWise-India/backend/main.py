
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "v3_1_random_forest_final.joblib"


# ==========================================
# 2. LOAD FINAL MODEL
# ==========================================

try:
    model = joblib.load(MODEL_PATH)
    print("CarWise V3.1 model loaded successfully!")

except Exception as e:
    model = None
    print(f"Model loading failed: {e}")


# ==========================================
# 3. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="CarWise India API",
    description="Used Car Price Prediction API using Machine Learning",
    version="3.1.0"
)


# ==========================================
# 4. CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 5. INPUT DATA VALIDATION
# ==========================================

class CarDetails(BaseModel):

    brand: str
    car_model: str
    variant_details: str

    car_age: float = Field(ge=0)
    km_driven: float = Field(ge=0)

    mileage: float = Field(gt=0)
    engine: float = Field(gt=0)
    max_power: float = Field(gt=0)
    seats: int = Field(gt=0)

    fuel: str
    seller_type: str
    transmission: str
    owner: str


# ==========================================
# 6. ROOT ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Welcome to CarWise India API",
        "version": "3.1.0",
        "model": "Random Forest",
        "status": "Running"
    }


# ==========================================
# 7. HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/health")
def health_check():

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Machine learning model is not loaded"
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": "V3.1"
    }


# ==========================================
# 8. PRICE PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict_price(car: CarDetails):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable"
        )

    try:

        # Convert input into DataFrame
        input_data = pd.DataFrame([{
            "car_age": car.car_age,
            "km_driven": car.km_driven,
            "mileage": car.mileage,
            "engine": car.engine,
            "max_power": car.max_power,
            "seats": car.seats,

            "brand": car.brand,
            "car_model": car.car_model,
            "variant_details": car.variant_details,

            "fuel": car.fuel,
            "seller_type": car.seller_type,
            "transmission": car.transmission,
            "owner": car.owner
        }])

        # Predict selling price
        prediction = model.predict(input_data)[0]

        predicted_price = max(0, float(prediction))

        return {
            "success": True,
            "model_version": "V3.1",
            "algorithm": "Random Forest",
            "predicted_price": round(predicted_price, 2),
            "currency": "INR",
            "formatted_price": f"₹{predicted_price:,.0f}"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
