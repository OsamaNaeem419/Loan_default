from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI()


# -------------------------------
# Load pipeline ONCE (important fix)
# -------------------------------
pipeline = PredictPipeline()


# -------------------------------
# Input Schema
# -------------------------------
class LoanInput(BaseModel):
    person_age: int
    person_income: float
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: int
    credit_score: int
    previous_loan_defaults_on_file: str


# -------------------------------
# Home Route
# -------------------------------
@app.get("/")
def home():
    return {"message": "Loan Prediction API Running"}


# -------------------------------
# Prediction Route
# -------------------------------
@app.post("/predict")
def predict(data: LoanInput):

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        # IMPORTANT FIX: enforce feature order
        expected_columns = [
            "person_age",
            "person_income",
            "person_emp_exp",
            "person_home_ownership",
            "loan_amnt",
            "loan_intent",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "credit_score",
            "previous_loan_defaults_on_file"
        ]

        input_df = input_df[expected_columns]

        # Prediction
        result = pipeline.predict(input_df)

        return {
            "prediction": result["prediction"],
            "probability": result["probability"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))