import requests
import pandas as pd

url = "http://127.0.0.1:8000/predict"

# -------------------------------
# TEST CASES (from your dataset)
# -------------------------------
cases = [
    {
        "name": "Case 1",
        "data": {
            "person_age": 22,
            "person_income": 71948,
            "person_emp_exp": 0,
            "person_home_ownership": "RENT",
            "loan_amnt": 35000,
            "loan_intent": "PERSONAL",
            "loan_int_rate": 16.02,
            "loan_percent_income": 0.49,
            "cb_person_cred_hist_length": 3,
            "credit_score": 561,
            "previous_loan_defaults_on_file": "No"
        }
    },
    {
        "name": "Case 2",
        "data": {
            "person_age": 21,
            "person_income": 12282,
            "person_emp_exp": 0,
            "person_home_ownership": "OWN",
            "loan_amnt": 1000,
            "loan_intent": "EDUCATION",
            "loan_int_rate": 11.14,
            "loan_percent_income": 0.08,
            "cb_person_cred_hist_length": 2,
            "credit_score": 504,
            "previous_loan_defaults_on_file": "Yes"
        }
    },
    {
        "name": "Case 3",
        "data": {
            "person_age": 25,
            "person_income": 12438,
            "person_emp_exp": 3,
            "person_home_ownership": "MORTGAGE",
            "loan_amnt": 5500,
            "loan_intent": "MEDICAL",
            "loan_int_rate": 12.87,
            "loan_percent_income": 0.44,
            "cb_person_cred_hist_length": 3,
            "credit_score": 635,
            "previous_loan_defaults_on_file": "No"
        }
    }
]

# -------------------------------
# RUN TESTS
# -------------------------------
results = []

for case in cases:
    print("\n========================")
    print(case["name"])
    print("Input:", case["data"])

    response = requests.post(url, json=case["data"])

    if response.status_code == 200:
        result = response.json()
        print("Output:", result)

        results.append({
            "case": case["name"],
            "prediction": result["prediction"],
            "probability": result["probability"]
        })
    else:
        print("Error:", response.text)

# -------------------------------
# SUMMARY TABLE
# -------------------------------
print("\n========================")
print("FINAL SUMMARY")
print("========================")

df = pd.DataFrame(results)
print(df)