from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
from app.model_utils import prepare_data, train_and_forecast

app = FastAPI(
    title="Angora XGBoost Forecast API",
    description="Forecast construction costs using XGBoost regression",
    version="1.0.0"
)

class ForecastRequest(BaseModel):
    data: list   # list of JSON records
    years: int = 5

@app.get("/")
def root():
    return {"status": "ok", "message": "Angora XGBoost Forecast API is running"}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    try:
        df = pd.DataFrame(req.data)
        df = prepare_data(df)
        forecast = train_and_forecast(df, req.years)
        return {
            "model": "XGBoost",
            "forecast": forecast,
            "training_years": len(df["year"].unique())
        }
    except Exception as e:
        return {"error": str(e)}
