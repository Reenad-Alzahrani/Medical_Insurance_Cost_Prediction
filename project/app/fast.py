from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from enum import Enum
from project.ml_logic.registry import load_model
from project.ml_logic.preprocessing import preprocessing


class Sex(str, Enum): female = "Female"; male = "Male"; other = "Other"
class Region(str, Enum): north = "North"; central = "Central"; west = "West"; south = "South"; east = "East"
class Education(str, Enum): doctorate = "Doctorate"; no_hs = "No HS"; hs = "HS"; some_college = "Some College"; masters = "Masters"; bachelors = "Bachelors"
class Employment(str, Enum): retired = "Retired"; employed = "Employed"; self_employed = "Self-employed"; unemployed = "Unemployed"
class Smoker(str, Enum): never = "Never"; current = "Current"; former = "Former"
class Tier(str, Enum): bronze = "Bronze"; gold = "Gold"; platinum = "Platinum"; silver = "Silver"


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

model = load_model(local_registry_path='model', model_name="GradientBoostin-model")
app.state.model = model

def align_to_schema(input_df: pd.DataFrame, full_schema: list) -> pd.DataFrame:
    aligned = pd.DataFrame([{col: 0.0 for col in full_schema}])
    for col in input_df.columns:
        if col in aligned.columns:
            aligned.at[0, col] = input_df.at[0, col]
    return aligned


@app.get("/predict")
def predict(
    age: int, sex: Sex, region: Region, bmi: float, children: int, smoker: Smoker,
    annual_premium: float, network_tier: Tier, deductible: float, total_claims_paid: float,
    avg_claim_amount: float, hba1c: float, provider_quality: float, diastolic_bp: float,
    income: float, education: Education, employment_status: Employment,
    medication_count: int, risk_score: float, diabetes: bool, arthritis: bool
):
    model = app.state.model
    if model is None:
        raise RuntimeError("Model failed to load")

    # Create a complete input dictionary with all required fields
    input_data = {
        'person_id': None,
        'age': age,
        'sex': sex.value,
        'region': region.value,
        'urban_rural': None,
        'income': income,
        'education': education.value,
        'marital_status': None,
        'employment_status': employment_status.value,
        'household_size': None,
        'dependents': None,
        'bmi': bmi,
        'smoker': smoker.value,
        'alcohol_freq': None,
        'visits_last_year': None,
        'hospitalizations_last_3yrs': None,
        'days_hospitalized_last_3yrs': None,
        'medication_count': medication_count,
        'systolic_bp': None,
        'diastolic_bp': diastolic_bp,
        'ldl': None,
        'hba1c': hba1c,
        'plan_type': None,
        'network_tier': network_tier.value,
        'deductible': deductible,
        'copay': None,
        'policy_term_years': None,
        'policy_changes_last_2yrs': None,
        'provider_quality': provider_quality,
        'risk_score': risk_score,
        'annual_premium': annual_premium,
        'monthly_premium': None,
        'claims_count': None,
        'avg_claim_amount': avg_claim_amount,
        'total_claims_paid': total_claims_paid,
        'chronic_count': None,
        'hypertension': None,
        'diabetes': diabetes,
        'asthma': None,
        'copd': None,
        'cardiovascular_disease': None,
        'cancer_history': None,
        'kidney_disease': None,
        'liver_disease': None,
        'arthritis': arthritis,
        'mental_health': None,
        'proc_imaging_count': None,
        'proc_surgery_count': None,
        'proc_physio_count': None,
        'proc_consult_count': None,
        'proc_lab_count': None,
        'is_high_risk': None,
        'had_major_procedure': None
    }

    input_df = pd.DataFrame([input_data])
    print("Input DataFrame:")
    print(input_df)

    input_data_preprocessed = preprocessing(input_df)
    print("Input data_preprocessed:")
    print(input_data_preprocessed)

    full_feature_list = model.feature_names_in_
    input_data_aligned = align_to_schema(input_data_preprocessed, full_feature_list)
    input_data_aligned = input_data_aligned.fillna(0.0)


    prediction = model.predict(input_data_aligned)

    return {"Predicted Annual Medical Cost": round(prediction[0], 2)}


@app.get("/")
def root():
    return dict(Project="Welcome to Medical_Insurance_Cost_Prediction_API")
