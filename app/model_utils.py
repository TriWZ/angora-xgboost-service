import pandas as pd
import numpy as np
import xgboost as xgb

def prepare_data(df: pd.DataFrame):
    # Normalize key columns
    date_col = "filing_date" if "filing_date" in df.columns else "approved_date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = df[date_col].dt.year
    if "estimated_cost" in df.columns:
        df["cost"] = pd.to_numeric(df["estimated_cost"].fillna(df.get("initial_cost")), errors="coerce")
    elif "cost" not in df.columns:
        raise ValueError("No 'cost' or 'estimated_cost' field found in data.")
    yearly = df.groupby("year")["cost"].mean().reset_index()
    return yearly.dropna()

def train_and_forecast(df: pd.DataFrame, years: int = 5):
    X = df[["year"]]
    y = df["cost"]

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.1)
    model.fit(X, y)

    last_year = int(df["year"].max())
    future_years = np.arange(last_year + 1, last_year + 1 + years)
    preds = model.predict(pd.DataFrame({"year": future_years}))

    forecast = [{"year": int(y), "predicted_cost": float(p)} for y, p in zip(future_years, preds)]
    return forecast
