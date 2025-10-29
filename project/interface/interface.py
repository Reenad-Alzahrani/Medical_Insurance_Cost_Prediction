import streamlit as st

import requests

with st.form(key='params_for_api'):

    st.subheader("Medical Insurance Cost Prediction")

    # Basic demographics
    age = st.number_input("Age", min_value=0, max_value=120, value=35)
    sex = st.selectbox("Sex", options=["male", "female"])
    region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.5)
    children = st.number_input("Children", min_value=0, max_value=10, value=2)
    smoker = st.selectbox("Smoker", options=[True, False])

    # High-importance features
    annual_premium = st.number_input("Annual Premium", min_value=0.0, value=12000.0)
    network_tier = st.selectbox("Network Tier", options=[1, 2, 3])
    deductible = st.number_input("Deductible", min_value=0.0, value=500.0)
    total_claims_paid = st.number_input("Total Claims Paid", min_value=0.0, value=3000.0)
    avg_claim_amount = st.number_input("Average Claim Amount", min_value=0.0, value=500.0)
    hba1c = st.number_input("HbA1c", min_value=4.0, max_value=15.0, value=5.6)
    provider_quality = st.number_input("Provider Quality", min_value=0.0, max_value=1.0, value=0.75)
    diastolic_bp = st.number_input("Diastolic BP", min_value=40.0, max_value=120.0, value=80.0)

    # Additional features
    income = st.number_input("Income", min_value=0.0, value=55000.0)
    education = st.selectbox("Education", options=["high_school", "bachelor", "master", "doctorate"])
    employment_status = st.selectbox("Employment Status", options=["employed", "unemployed", "retired", "student"])
    medication_count = st.number_input("Medication Count", min_value=0, max_value=20, value=2)
    risk_score = st.number_input("Risk Score", min_value=0.0, max_value=1.0, value=0.3)
    diabetes = st.selectbox("Diabetes", options=[True, False])
    arthritis = st.selectbox("Arthritis", options=[True, False])
    submit_button = st.form_submit_button(label='Predict Cost')

    if submit_button:
        input_data = {
            "age": age,
            "sex": sex,
            "region": region,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "annual_premium": annual_premium,
            "network_tier": network_tier,
            "deductible": deductible,
            "total_claims_paid": total_claims_paid,
            "avg_claim_amount": avg_claim_amount,
            "hba1c": hba1c,
            "provider_quality": provider_quality,
            "diastolic_bp": diastolic_bp,
            "income": income,
            "education": education,
            "employment_status": employment_status,
            "medication_count": medication_count,
            "risk_score": risk_score,
            "diabetes": diabetes,
            "arthritis": arthritis
        }

        # Replace with your actual FastAPI endpoint
        response = requests.get("http://localhost:8000/predict", params=input_data)

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Predicted Annual Medical Cost: ${prediction[0]:,.2f}")
        else:
            st.error("Prediction failed. Check API or input values.")
