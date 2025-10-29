
import os
import time
import pandas as pd
import numpy as np
from colorama import Fore, Style
from project.ml_logic.registry import save_model

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from joblib import dump

from project.ml_logic.preprocessing import preprocessing

DATA_PATH = 'raw_data/medical_insurance.csv'

print(Fore.BLUE + "\n Loading dataset..." + Style.RESET_ALL)
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=['annual_medical_cost'])
y = df['annual_medical_cost']

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(Fore.GREEN + "✅ Data loaded and split successfully" + Style.RESET_ALL)


def evaluate_model(name, model, X_test, y_test):

    """Evaluate model performance and print results"""

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(Fore.CYAN + f"\n{name} evaluated metrics:" + Style.RESET_ALL)
    print(f"  ➤ MAE = {mae:.2f}")
    print(f"  ➤ R²  = {r2:.3f}")
    return [name, mae, r2]


# Gradient Boosting
print(Fore.BLUE + "\nTraining Gradient Boosting model..." + Style.RESET_ALL)

gbr = GradientBoostingRegressor(max_depth=10,
                                n_estimators=300,
                                learning_rate=0.1)

pipe_gbr = make_pipeline(FunctionTransformer(preprocessing), gbr)

start = time.time()
X_train = preprocessing(X_train)
gbr.fit(X_train, y_train)
end = time.time()

print(Fore.GREEN + f"Model trained successfully in {end - start:.2f} seconds" + Style.RESET_ALL)


# evaluate the model
X_test = preprocessing(X_test)
evaluate_model("Gradient Boosting Regressor", gbr, X_test, y_test)

# cv_results = cross_validate(pipe_gbr, X_train, y_train, scoring="r2", cv=3, return_train_score=True)

# print(Fore.YELLOW + f"\nCross-validation results:" + Style.RESET_ALL)
# print(f"  ➤ Train R² Mean: {cv_results['train_score'].mean():.3f}")
# print(f"  ➤ Test  R² Mean: {cv_results['test_score'].mean():.3f}")


# save the model

save_model(gbr, local_registry_path='model',model_name="GradientBoostin-model")

# os.makedirs("model", exist_ok=True)
# MODEL_PATH = "model/gradient_boosting_model.pkl"
# dump(pipe_gbr, MODEL_PATH)
# print(Fore.MAGENTA + f"\n Model saved successfully to" + Style.RESET_ALL)
