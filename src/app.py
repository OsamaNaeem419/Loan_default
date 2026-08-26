from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI(title="Loan Approval API (two-stage)")

# ---------------------------------------------------------------------------
# CORS — the React dev server runs on a different origin (port 5173), so the
# browser needs this to be allowed to call /predict. Tighten allow_origins to
# the deployed frontend URL before going to production.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pipeline once at startup.
pipeline = PredictPipeline()


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
    # Used by the Stage-1 policy gate, not by the model.
    previous_loan_defaults_on_file: str


@app.get("/")
def home():
    return {"message": "Loan Prediction API running (two-stage)"}


@app.post("/predict")
def predict(data: LoanInput):
    """
    Returns the decision plus which stage produced it:
      - stage = "policy_gate": rejected by the prior-default rule; `reason` set.
      - stage = "model":       scored by Stage 2; `explanation` + `counterfactual` set.
    """
    try:
        input_df = pd.DataFrame([data.model_dump()])
        result = pipeline.predict(input_df)
        return {
            "prediction": result["prediction"],
            "probability": result["probability"],
            "stage": result["stage"],
            "reason": result["reason"],
            "explanation": result["explanation"],
            "counterfactual": result["counterfactual"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
