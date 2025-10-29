import streamlit as st
import requests
import pandas as pd
from project.app.fast import Sex, Region, Education, Employment, MaritalStatus, Smoker, Plan, Tier


st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="💊",
    layout="wide"
)

st.markdown("""
    <style>
    body {
        background-color: lightblue;
        color: #003366;
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background-color: lightblue;
    }
    .title-container {
        background-color: #b3e5fc;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0px 4px 12px rgba(0, 102, 153, 0.2);
    }
    .title-container h1 {
        color: #003366;
        font-weight: 800;
        font-size: 36px;
        margin-bottom: 10px;
    }
    .title-container p {
        color: #004d80;
        font-size: 16px;
        font-weight: 500;
    }
    .stButton > button {
        background-color: #4fc3f7;
        color: #003366;
        font-weight: 600;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #29b6f6;
        transform: scale(1.05);
    }
    .card {
        background-color: #dff6ff;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        flex: 1;
        border: 1px solid #b3e5fc;
        color: darkblue ;
        box-shadow: 0px 2px 8px rgba(0, 128, 255, 0.1);
    }
    .card h4 {
        margin-bottom: 5px;
        color: #004080;
        font-size: 16px;
        font-weight: 600;
    }
    .card h2 {
        color: #0074D9;
        margin-top: 0;
        font-size: 22px;
    }
    .prediction-box {
        background-color: #d9f0ff;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        border: 2px solid #b3e5fc;
    }
    .prediction-box h3 {
        color: #003366;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Styling ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2f 0%, #151520 100%);
        color: #FFFFFF;
        padding: 1.5rem 1rem 2rem 1rem;
        border-right: 2px solid #4FC3F7;
    }
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #4FC3F7;
        text-align: left;
        margin-bottom: 0.5rem;
    }
    .sidebar-sub {
        font-size: 0.85rem;
        color: #cccccc;
        text-align: left;
        margin-bottom: 1.5rem;
    }
    .sidebar-section {
        background-color: #262639;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .sidebar-section h3 {
        color: #FFD54F;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .sidebar-section p {
        color: #cccccc;
        font-size: 0.9rem;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Content ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=90)

st.sidebar.markdown('<div class="sidebar-title">🏥 PredictCare: Your Health Cost Navigator</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-sub">Unlock smarter decisions with ML-powered predictions for your annual medical insurance costs.</div>', unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-section">
    <h3>⚙️ How It Works</h3>
    <p>Share your personal, medical, and insurance details. Our machine learning model uses real-world data to predict your annual healthcare costs—helping you plan with confidence.</p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="title-container">
    <h1>🏥 PredictCare: Smarter Insurance Cost Estimation</h1>
    <p>Harness the power of machine learning to forecast your personalized medical insurance expenses with confidence.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 5, 1])
with col2:
        st.subheader("👋 Let’s Personalize Your Prediction — You Share, We Calculate")

with st.form(key='input_form'):
    col1, col2, col3 = st.columns([1.2, 5, 1])
    with col2:
        st.markdown("Ready to see your forecast? Fill in the details and let PredictCare do the rest.")

    form_col1, form_col2, form_col3 = st.columns(3)

    with form_col1:
        st.markdown("#### 🧍 Personal & Lifestyle")
        age = st.number_input("Age", min_value=0, max_value=120, value=35)
        sex = st.selectbox("Gender", options=[s.value for s in Sex])
        marital_status = st.selectbox("Marital Status", options=[s.value for s in MaritalStatus])
        household_size = st.number_input("Total family members", min_value=1, max_value=20, value=4)
        smoker = st.selectbox("Are you a smoker?", options=[s.value for s in Smoker])
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=27.5)
        region = st.selectbox("Region", options=[s.value for s in Region])
        education = st.selectbox("Education", options=[s.value for s in Education])
        employment_status = st.selectbox("Employment Status", options=[s.value for s in Employment])
        income = st.number_input("Income", min_value=0.0, value=55000.0)
    with form_col2:
        st.markdown("#### 🩺 Medical History")
        visits_last_year = st.number_input("Doctor/clinic visits (past year)", min_value=0, max_value=50, value=3)
        hospitalizations_last_3yrs = st.number_input("Hospital stays (last 3 years)", min_value=0, max_value=10, value=1)
        diabetes = st.selectbox("Do you have diabetes?", options=[True, False])
        arthritis = st.selectbox("Diagnosed with arthritis?", options=[True, False])
        medication_count = st.number_input("Medications you're taking", min_value=0, max_value=20, value=2)
        provider_quality = st.number_input("Provider quality rating", min_value=0.0, max_value=5.0, value=3.0)
        hba1c = st.number_input("A1c % (last 3 months)", min_value=4.0, max_value=15.0, value=5.6)
        diastolic_bp = st.number_input("Diastolic BP (bottom)", min_value=40.0, max_value=120.0, value=80.0)
        systolic_bp = st.number_input("Systolic BP (top)", min_value=80.0, max_value=200.0, value=120.0)
        ldl = st.number_input("LDL cholesterol level", min_value=50.0, max_value=300.0, value=130.0)
        proc_physio_count = st.number_input("Psychiatric sessions", min_value=0, max_value=20, value=1)
        proc_consult_count = st.number_input("Specialist visits", min_value=0, max_value=20, value=2)

    with form_col3:
        st.markdown("#### 💳 Insurance Details")
        annual_premium = st.number_input("Annual insurance premium", min_value=0.0, value=12000.0)
        deductible = st.number_input("Deductible amount", min_value=0.0, value=5000.0)
        claims_count = st.number_input("Claims submitted (last year)", min_value=0, max_value=50, value=5)
        total_claims_paid = st.number_input("Total claims paid by insurer", min_value=0.0, value=3000.0)
        plan_type = st.selectbox("Insurance plan type", options=[s.value for s in Plan])
        network_tier = st.selectbox("Insurance plan level", options=[s.value for s in Tier])

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
         submit_button = st.form_submit_button(label='🚀 Predict My Insurance Cost')

col1, col2, col3 = st.columns([2, 4, 1])
with col2:
        st.subheader("📊 Your Personalized Cost Forecast")

if 'prediction' not in st.session_state:
        st.session_state.prediction = None

if submit_button:
        input_data = {"age": age,
                    "sex": sex,
                    "region": region,
                    "education": education,
                    "employment_status": employment_status,
                    "marital_status": marital_status,
                    "household_size": household_size,
                    "income": income,
                    "bmi": bmi,
                    "smoker": smoker,
                    "visits_last_year": visits_last_year,
                    "hospitalizations_last_3yrs": hospitalizations_last_3yrs,
                    "deductible": deductible,
                    "claims_count": claims_count,
                    "total_claims_paid": total_claims_paid,
                    "provider_quality": provider_quality,
                    "hba1c": hba1c,
                    "diastolic_bp": diastolic_bp,
                    "systolic_bp": systolic_bp,
                    "ldl": ldl,
                    "diabetes": diabetes,
                    "medication_count": medication_count,
                    "proc_physio_count": proc_physio_count,
                    "proc_consult_count": proc_consult_count,
                    "arthritis": arthritis,
                    "annual_premium": annual_premium,
                    "plan_type": plan_type,
                    "network_tier": network_tier}


        response = requests.get("http://localhost:8000/predict", params=input_data)

        if response.status_code == 200:
            result = response.json()
            st.session_state.prediction = result.get("Predicted Annual Medical Cost", 0)
        else:
            st.error("⚠️ Prediction failed. Check API connection.")

        if st.session_state.prediction is not None:
            st.markdown(f"""
            <div class="prediction-box" style="background-color:#f0f8ff;padding:20px;border-radius:10px;">
                <h3 style="margin-bottom:10px;">💰 Your Estimated Annual Medical Cost</h3>
                <h1 style="color:#0066cc;font-size:36px;">${st.session_state.prediction:,.2f}</h1>
                <p style="font-size:14px;color:#555;">Based on your profile, this is the expected yearly expense for medical insurance.</p>
            </div>
            """, unsafe_allow_html=True)


            st.markdown(f"""
            <div style='display:flex;gap:20px;margin-top:30px;flex-wrap:wrap;'>
                <div class='card' style='flex:1;padding:15px;border-radius:10px;background-color:#F5F5F5;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.1);'>
                    <h4 style='margin-bottom:10px;'>🏥Insurance plan</h4>
                    <h5 style='color:#1E88E5;'>{plan_type}</h5>
                </div>
                <div class='card' style='flex:1;padding:15px;border-radius:10px;background-color:#FFF3E0;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.1);'>
                    <h4 style='margin-bottom:10px;'>🧮Insurance plan level</h4>
                    <h2 style='color:#EF6C00;'>{network_tier}</h2>
                </div>
                <div class='card' style='flex:1;padding:15px;border-radius:10px;background-color:#FFF3E0;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.1);'>
                    <h4 style='margin-bottom:10px;'>💰Total paid before insurance</h4>
                    <h3 style='color:#2E7D32;'>{deductible}$</h3>
                </div>
            </div>
            """, unsafe_allow_html=True)


            st.markdown("""
                        <br><hr>
                        <h4 style='text-align:center;color:#004d80;font-size:24px;'>🩺 Key Health & Financial Insights</h4>
                        """, unsafe_allow_html=True)

            col_health, col_finance, col_personal = st.columns(3)

            with col_health:
                st.markdown(f"""
                 <div class='card' style='background-color:#E3F2FD;padding:15px;border-radius:10px;box-shadow:0 2px 5px rgba(0,0,0,0.1);'>
                <h4 style='color:#1565C0;'>💪 Health Indicators</h4>
                <p><b>BMI:</b> {bmi:.1f}</p>
                <p><b>Average blood sugar:</b> {hba1c:.1f}</p>
                <p><b>LDL level (bad cholesterol):</b> {ldl:.1f}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_finance:
                st.markdown(f"""
                <div class='card' style='background-color:#E8F5E9;padding:15px;border-radius:10px;box-shadow:0 2px 5px rgba(0,0,0,0.1);'>
                <h4 style='color:#2E7D32;'>💰 Financial Overview</h4>
                <p><b>Income:</b> ${income:,.0f}</p>
                <p><b>Premium:</b> ${annual_premium:,.0f}</p>
                <p><b>claims paid by insurer:</b> ${total_claims_paid:,.0f}</p>
                 </div>
                """, unsafe_allow_html=True)

            with col_personal:
                st.markdown(f"""
                <div class='card' style='background-color:#FFF3E0;padding:15px;border-radius:10px;box-shadow:0 2px 5px rgba(0,0,0,0.1);'>
                <h4 style='color:#EF6C00;'>👤 Personal Details</h4>
                <p><b>Age:</b> {age} years</p>
                <p><b>Gender:</b> {sex}</p>
                <p><b>Education:</b> {education}</p>
                 </div>
                """, unsafe_allow_html=True)
