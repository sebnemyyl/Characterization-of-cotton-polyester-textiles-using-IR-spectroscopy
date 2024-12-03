from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor  # Import MLPRegressor for neural network
from sklearn.svm import SVR

import pandas as pd
import numpy as np
import time

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
    #"XGBoost": RandomizedSearchCV(
    #    XGBRegressor(),
    #    param_distributions={
    #        "n_estimators": [100, 200, 300],
    #        "max_depth": [3, 5, 7],
    #        "learning_rate": [0.01, 0.1, 0.2],
    #        "subsample": [0.8, 0.9, 1.0]
    #    },
    #    n_iter=10, random_state=42
    #),
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

def evaluate_model(model_name, X_train, X_test, y_train, y_test):
    model = models[model_name]
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

def load_feature_set(csv_file):
    baseline_corrected_data = pd.read_csv(csv_file, sep=',', header=0)
    # Convert the cotton column to numeric and handle errors
    baseline_corrected_data['reference.cotton'] = pd.to_numeric(baseline_corrected_data['reference.cotton'], errors='coerce')
    # Drop rows with missing cotton values
    data_clean = baseline_corrected_data.dropna(subset=['reference.cotton'])

    # Prepare the feature set (exclude non-spectral and cotton columns)
    X = data_clean.drop(columns=['Unnamed: 0', 'reference.pet', 'reference.cotton', 'reference.specimen', 'reference.area', 'reference.spot', 'reference.measuring_date'])
    X.columns = X.columns.str.replace('spectra.', '')
    # Prepare the target column (cotton content)
    y = data_clean['reference.cotton']

    return (X, y)