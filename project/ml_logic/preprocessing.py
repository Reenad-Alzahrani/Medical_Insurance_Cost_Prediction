from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd



def simpleimputer(data):
    """
    Fill missing values for all columns in the dataframe
    using the most frequent value (mode) for each column.
    """
    imputer = SimpleImputer(strategy='most_frequent')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data_imputed

    #imputer= SimpleImputer(strategy='most_frequent')
    #data['alcohol_freq'] = imputer.fit_transform(data[['alcohol_freq']])[:,0]
    #return data

def normalize_column_names(data):
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
    return data


def standardscaler(data):
    standard_cols = ['systolic_bp', 'diastolic_bp', 'ldl', 'bmi',
                     'age', 'provider_quality']
    scaler_standard = StandardScaler()
    df_standard_scaled = data.copy()
    df_standard_scaled[standard_cols] = scaler_standard.fit_transform(data[standard_cols])
    return df_standard_scaled

def minmaxscaler(data):
    minmax_cols = ['total_claims_paid', 'avg_claim_amount', 'policy_changes_last_2yrs',
                'annual_premium','days_hospitalized_last_3yrs',
                'proc_surgery_count', 'income', 'hospitalizations_last_3yrs', 'hba1c',
                'deductible', 'claims_count', 'proc_consult_count', 'proc_lab_count',
                'proc_imaging_count', 'proc_physio_count', 'visits_last_year',
                'medication_count', 'dependents', 'chronic_count', 'household_size']
    scaler_minmax = MinMaxScaler()
    df_minmax_scaled = data.copy()
    df_minmax_scaled[minmax_cols] = scaler_minmax.fit_transform(data[minmax_cols])
    return df_minmax_scaled

def onehotencoder(data):
    encoder_columns = ['sex', 'region', 'urban_rural', 'education', 'marital_status',
                       'employment_status', 'smoker', 'alcohol_freq', 'plan_type']
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_array = ohe.fit_transform(data[encoder_columns])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=ohe.get_feature_names_out(encoder_columns),
        index=data.index)
    return pd.concat(
        [data.drop(columns=encoder_columns, errors='ignore'), encoded_df],
        axis=1)



def ordinalencoder(data):
    tier_order = [['Bronze', 'Silver', 'Gold', 'Platinum']]
    ode = OrdinalEncoder(categories=tier_order)
    return pd.concat([data.drop(columns=['network_tier'], errors='ignore'),
                      pd.DataFrame(ode.fit_transform(data[['network_tier']]),
                      columns=ode.get_feature_names_out(['network_tier']),
                      index=data.index)], axis=1)

def preprocessing(data):
    data = data.drop(columns=['monthly_premium', 'person_id'])
    data = simpleimputer(data)
    data = standardscaler(data)
    data = minmaxscaler(data)
    data = ordinalencoder(data)
    data = onehotencoder(data)
    data = normalize_column_names(data)
    return data



# test if the code works
if __name__ == "__main__":
    # test the preprocessing function
    df = pd.read_csv("raw_data/medical_insurance.csv")
    X = df.drop(columns=['annual_medical_cost'])
    y = df['annual_medical_cost']
    print("Before Preprocessing:\n", X.head())
    df_preprocessed = preprocessing(X)
    print("After Preprocessing:\n", df_preprocessed.head())
