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
│   ├── models/
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


## How to Run

### 1. Start FastAPI backend
uvicorn main:app --reload

### 2. Start Streamlit frontend
streamlit run webapp/streamlit_app.py

---


## Business Logic

- Prediction = 1 → Loan Approved
- Prediction = 0 → Loan Rejected
- Probability represents model confidence in approval

---

