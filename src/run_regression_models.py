from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from xgboost import DMatrix, train, cv  # Use xgboost as xgb
from sklearn.neural_network import MLPRegressor  # Import MLPRegressor for neural network


def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Predict
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "best_params": model.best_params_,
        "training_time": training_time,
        "prediction_time": prediction_time,
        "mse": mse,
        "r2": r2
    }

# Load the dataset
print(os.getcwd())
my_path = "../input"
file = "spectra_nir_all.csv"
baseline_corrected_data = pd.read_csv(f'{my_path}/{file}', sep=',', header=0)

# Convert the cotton column to numeric and handle errors
baseline_corrected_data['reference.cotton'] = pd.to_numeric(baseline_corrected_data['reference.cotton'], errors='coerce')

# Drop rows with missing cotton values
data_clean = baseline_corrected_data.dropna(subset=['reference.cotton'])

# Prepare the feature set (exclude non-spectral and cotton columns)
X = data_clean.drop(columns=['Unnamed: 0', 'reference.pet', 'reference.cotton', 'reference.specimen', 'reference.area', 'reference.spot', 'reference.measuring_date'])

X.columns = X.columns.str.replace('spectra.', '')

# Prepare the target column (cotton content)
y = data_clean['reference.cotton']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "SVR": RandomizedSearchCV(
        SVR(kernel="rbf"),
        param_distributions={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
        n_iter=10, random_state=42
    ),
    "Kernel Ridge": RandomizedSearchCV(
        KernelRidge(kernel="rbf"),
        param_distributions={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
        n_iter=10, random_state=42
    ),
    "XGBoost": RandomizedSearchCV(
        XGBRegressor(),
        param_distributions={
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 0.9, 1.0]
        },
        n_iter=10, random_state=42
    ),
    "MLP": RandomizedSearchCV(
        MLPRegressor(max_iter=1000),
        param_distributions={
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'activation': ['relu', 'tanh']
        },
        n_iter=10, random_state=42
    )
}

# Train and evaluate all models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    print(f"{name} done. Best params: {results[name]['best_params']}")

# Display results
for name, res in results.items():
    print(f"\n{name} Results:")
    print(f"  - R2 Score: {res['r2']:.3f}")
    print(f"  - MSE: {res['mse']:.3f}")
    print(f"  - Training Time: {res['training_time']:.3f}s")
    print(f"  - Prediction Time: {res['prediction_time']:.3f}s")




#
# # Cross-validate SVR
# t0 = time.time()
# svr_cv_scores = cross_val_score(svr, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
# svr_fit = time.time() - t0
# print(f"SVR cross-validation MSE: {-svr_cv_scores.mean():.3f} (+/- {svr_cv_scores.std():.3f})")
# print(f"SVR fitted in %.3f s" % svr_fit)
#
# # Cross-validate Kernel Ridge Regression
# t0 = time.time()
# kr_cv_scores = cross_val_score(kr, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
# kr_fit = time.time() - t0
# print(f"KRR cross-validation MSE: {-kr_cv_scores.mean():.3f} (+/- {kr_cv_scores.std():.3f})")
# print(f"KRR fitted in %.3f s" % kr_fit)


