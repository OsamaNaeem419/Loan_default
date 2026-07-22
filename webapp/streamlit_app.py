import streamlit as st
import requests

st.title("Loan Approval Prediction System")

# -------------------------------
# INPUT UI
# -------------------------------
st.subheader("Enter Loan Details")

col1, col2, col3 = st.columns(3)

# NOTE: person_gender and person_education were removed from this form --
# the training pipeline (data_ingestion.py) explicitly drops both columns
# before the model ever sees them, so the model was never trained on
# gender or education. Asking the user for them did nothing except add
# two pointless fields to fill in.
with col1:
    person_age = st.number_input("Age", 18, 100, 25)
    person_emp_exp = st.number_input("Work Experience", 0, 50, 0)

with col2:
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    cb_person_cred_hist_length = st.number_input("Credit History Length", 0, 50, 3)

with col3:
    person_income = st.number_input("Income", value=50000)
    loan_amnt = st.number_input("Loan Amount", value=10000)

col4, col5 = st.columns(2)

with col4:
    loan_int_rate = st.number_input("Interest Rate", value=10.0)

with col5:
    # loan_percent_income is just loan_amnt / person_income -- it's fully
    # determined by the two fields above, so it shouldn't be a free-form
    # input the user can set to something inconsistent. Compute it
    # automatically instead and just show it to them.
    loan_percent_income = round(loan_amnt / person_income, 4) if person_income > 0 else 0.0
    st.metric("Loan % of Income", f"{loan_percent_income:.2f}")

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