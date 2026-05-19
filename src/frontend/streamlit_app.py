import streamlit as st
import requests

st.title("Loan Approval Prediction System")

# -------------------------------
# INPUT UI
# -------------------------------
st.subheader("Enter Loan Details")

col1, col2, col3 = st.columns(3)

with col1:
    person_age = st.number_input("Age", 18, 100, 25)
    person_gender = st.selectbox("Gender", ["male", "female"])

with col2:
    person_education = st.selectbox(
        "Education",
        ["High School", "Associate", "Bachelor", "Master"]
    )
    person_emp_exp = st.number_input("Work Experience", 0, 50, 0)

with col3:
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    cb_person_cred_hist_length = st.number_input("Credit History Length", 0, 50, 3)

col4, col5 = st.columns(2)

with col4:
    person_income = st.number_input("Income", value=50000)
    loan_amnt = st.number_input("Loan Amount", value=10000)

with col5:
    loan_int_rate = st.number_input("Interest Rate", value=10.0)
    loan_percent_income = st.number_input("Loan % Income", value=0.2)

person_home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_intent = st.selectbox(
    "Loan Intent",
    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)

previous_loan_defaults_on_file = st.selectbox(
    "Previous Default",
    ["Yes", "No"]
)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict 🚀"):

    payload = {
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    if response.status_code == 200:
        result = response.json()

        prediction = result["prediction"]
        probability = result["probability"]

        st.subheader("Result")

        if prediction == 1:
            st.success("APPROVED")
        else:
            st.error("REJECTED")

        st.metric("Probability", f"{probability:.2f}")

    else:
        st.error("API Error")