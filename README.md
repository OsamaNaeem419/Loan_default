# Loan Approval Prediction System

## Overview

This project is an end-to-end machine learning system that predicts whether a loan application will be approved or rejected based on applicant financial and personal information.

The system includes:
- Data ingestion from database/CSV
- Data preprocessing & feature engineering
- Machine learning model training
- FastAPI backend for real-time prediction
- Streamlit frontend for user interaction

---

## Architecture

User Input (Streamlit UI)
        ↓
FastAPI Backend
        ↓
Preprocessing Pipeline
        ↓
ML Model (Prediction)
        ↓
Response (Probability + Decision)

---

## Features

- End-to-end ML pipeline
- Real-time predictions using FastAPI
- Interactive UI using Streamlit
- Probability-based decision output
- Threshold tuning for business control
- Handles categorical and numerical features

---

## Project Structure

loan-approval-system/
│
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   └── test.csv
│
├── web_app/
│   └── streamlit_app.py
│
├── main.py
├── requirements.txt
└── README.md

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Streamlit
- Pickle (Model Serialization)

---

## Model Details

- Model outputs probability of loan approval
- Decision threshold tuning:
  - 0.94 → Low risk (minimize false positives)
  - 0.69 → Balanced performance (used in this project)

Final deployed threshold: 0.69

---

## How to Run

### 1. Start FastAPI backend
uvicorn main:app --reload

### 2. Start Streamlit frontend
streamlit run web_app/streamlit_app.py

---

## API

POST /predict

Request:
{
  "person_age": 25,
  "person_gender": "male",
  "person_education": "Bachelor",
  "person_income": 50000,
  "person_emp_exp": 2,
  "person_home_ownership": "RENT",
  "loan_amnt": 10000,
  "loan_intent": "EDUCATION",
  "loan_int_rate": 10.5,
  "loan_percent_income": 0.2,
  "cb_person_cred_hist_length": 3,
  "credit_score": 650,
  "previous_loan_defaults_on_file": "No"
}

Response:
{
  "prediction": 1,
  "probability": 0.82
}

---

## Business Logic

- Prediction = 1 → Loan Approved
- Prediction = 0 → Loan Rejected
- Probability represents model confidence in approval

---

## Key Insight

Threshold tuning was used to balance:
- False Positives (bad loans approved)
- False Negatives (good loans rejected)

This makes the system adaptable to business risk requirements.

---

## Conclusion

This project demonstrates a full end-to-end machine learning pipeline with deployment-ready architecture using FastAPI and Streamlit.