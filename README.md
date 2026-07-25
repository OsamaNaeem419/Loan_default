# Loan Approval Prediction System

## Overview

This project is an end-to-end machine learning system that predicts whether a loan application will be approved or rejected based on applicant financial and personal information.

The system includes:
- Data ingestion from database/CSV
- Data preprocessing & feature engineering
- Machine learning model training
- FastAPI backend for real-time prediction
- React frontend for user interaction

---

## Architecture

User Input (React UI)
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
- Interactive UI using React (Vite)
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
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── requirements.txt
└── README.md

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- React + Vite
- Pickle (Model Serialization)

---


## How to Run

### 1. Start FastAPI backend
From the project root:

    pip install -r requirements.txt
    uvicorn src.app:app --reload --port 8000

Backend runs on http://127.0.0.1:8000 (interactive docs at /docs).

### 2. Start React frontend
In a second terminal:

    cd frontend
    npm install
    npm run dev

Frontend runs on http://localhost:5173

The frontend calls the backend at http://127.0.0.1:8000 by default. To point it
somewhere else, create `frontend/.env.local` with:

    VITE_API_URL=http://your-host:port

---


## Business Logic

- Prediction = 1 → Loan Approved
- Prediction = 0 → Loan Rejected
- Probability represents model confidence in approval

---

