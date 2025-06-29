# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib


class SalaryRequest(BaseModel):
    work_year: int = Field(..., example=2023)
    remote_ratio: float = Field(..., ge=0, le=100, example=100)
    experience_level: str = Field(..., example="SE")
    employment_type: str   = Field(..., example="FT")
    salary_currency: str   = Field(..., example="USD")
    employee_residence: str= Field(..., example="US")
    company_location: str  = Field(..., example="US")
    company_size: str      = Field(..., example="M")
    job_title: str         = Field(..., example="Data Scientist")


app = FastAPI(
    title="Salary Prediction API",
    description="Predicción de salario usando el pipeline de RandomForest",
    version="1.0"
)


pipeline = joblib.load("pipeline.joblib")

@app.post("/predict")
def predict_salary(req: SalaryRequest):
    df = pd.DataFrame([req.dict()])
    try:
        pred = pipeline.predict(df)
        return {"salary_prediction_usd": float(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

