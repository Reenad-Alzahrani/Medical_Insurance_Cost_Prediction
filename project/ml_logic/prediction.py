import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from project.ml_logic.preprocessing import preprocessing
from project.ml_logic.registry import load_model



# predict function
def get_default_value(col):
    default_map = {
        "person_id": "unknown", "urban_rural": "Urban", "education": "HS", "marital_status": "Single",
        "employment_status": "Employed", "household_size": 1, "dependents": 0, "smoker": "Never",
        "alcohol_freq": "Occasional", "visits_last_year": 1, "hospitalizations_last_3yrs": 0,
        "days_hospitalized_last_3yrs": 0, "medication_count": 0, "systolic_bp": 120, "ldl": 100,
        "plan_type": "PPO", "copay": 20, "policy_term_years": 1, "policy_changes_last_2yrs": 0,
        "annual_medical_cost": 0, "monthly_premium": 0, "claims_count": 0, "chronic_count": 0,
        "hypertension": 0, "asthma": 0, "copd": 0, "cardiovascular_disease": 0, "cancer_history": 0,
        "kidney_disease": 0, "liver_disease": 0, "mental_health": 0, "proc_imaging_count": 0,
        "proc_surgery_count": 0, "proc_physio_count": 0, "proc_consult_count": 0, "proc_lab_count": 0,
        "is_high_risk": 0, "had_major_procedure": 0
    }
    return default_map.get(col, 0)

def align_input_to_model(input_data: dict, expected_columns: list):
    df = pd.DataFrame([input_data])
    missing = [col for col in expected_columns if col not in df.columns]
    for col in missing:
        df[col] = get_default_value(col)
    df = df[expected_columns]
    print("✅ Missing columns filled:", missing)
    return df
