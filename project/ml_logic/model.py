
import os
import time
import pandas as pd
import numpy as np
from colorama import Fore, Style
from project.ml_logic.registry import save_model

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
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
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(Fore.CYAN + f"\n{name} evaluated metrics:" + Style.RESET_ALL)
    print(f"  ➤ RMSE = {rmse:.2f}")
    print(f"  ➤ R²   = {r2:.3f}")
    return [name, rmse, r2]


# Gradient Boosting
print(Fore.BLUE + "\nTraining Gradient Boosting model..." + Style.RESET_ALL)

gboost = GradientBoostingRegressor(n_estimators=30,learning_rate=0.05,max_depth=20,
                                   max_features=10,min_samples_split=15,
                                    subsample=0.6, verbose=0)

pipe_gboost = make_pipeline(FunctionTransformer(preprocessing), gboost)

start = time.time()
X_train = preprocessing(X_train)
gboost.fit(X_train, y_train)
end = time.time()

print(Fore.GREEN + f"Model trained successfully in {end - start:.2f} seconds" + Style.RESET_ALL)


# evaluate the model
X_test_preproc = preprocessing(X_test)
evaluate_model("Gradient Boosting Regressor", gboost, X_test_preproc, y_test)

cv_results = cross_validate(gboost, X_train, y_train, scoring="neg_root_mean_squared_error", cv=3, return_train_score=True)

print(Fore.YELLOW + f"\nCross-validation results:" + Style.RESET_ALL)
print(f"  ➤ Train R² Mean: {-cv_results['train_score'].mean():.3f}")
print(f"  ➤ Test  R² Mean: {-cv_results['test_score'].mean():.3f}")


# save the model

save_model(gboost, local_registry_path='model',model_name="GradientBoostin-model")
