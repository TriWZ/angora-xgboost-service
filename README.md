# Angora XGBoost Forecast API

This is a lightweight FastAPI microservice that forecasts future construction costs using **XGBoost regression**.

## 🚀 Endpoints

- `GET /` → Health check
- `POST /forecast` → Train and forecast
  - Input JSON:
    ```json
    {
      "data": [
        {"filing_date": "2019-01-01", "estimated_cost": 5000},
        {"filing_date": "2020-01-01", "estimated_cost": 6000}
      ],
      "years": 5
    }
    ```
  - Output:
    ```json
    {
      "model": "XGBoost",
      "forecast": [
        {"year": 2025, "predicted_cost": 7023.4},
        {"year": 2026, "predicted_cost": 7200.7}
      ]
    }
    ```

## 🧰 Local Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
