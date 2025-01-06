from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor  # Import MLPRegressor for neural network
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV

def load_feature_set(csv_file):
    baseline_corrected_data = pd.read_csv(csv_file, sep=',', header=0)
    # Convert the cotton column to numeric and handle errors
    baseline_corrected_data['reference.cotton'] = pd.to_numeric(baseline_corrected_data['reference.cotton'], errors='coerce')
    # Drop rows with missing cotton values
    data_clean = baseline_corrected_data.dropna(subset=['reference.cotton'])
    # Prepare the feature set (exclude non-spectral columns)
    X = data_clean.drop(columns=['Unnamed: 0', 'reference.pet', 'reference.cotton', 'reference.specimen', 'reference.area', 'reference.spot', 'reference.measuring_date'])
    X.columns = X.columns.str.replace('spectra.', '')
    # Prepare the target column (cotton content)
    y = data_clean['reference.cotton']
    return (X, y)

def split_feature_set_with_specimen(csv_file):
    baseline_corrected_data = pd.read_csv(csv_file, sep=',', header=0)
    # Convert the cotton column to numeric and handle errors
    baseline_corrected_data['reference.cotton'] = pd.to_numeric(baseline_corrected_data['reference.cotton'], errors='coerce')
    # Drop rows with missing cotton values
    data_clean = baseline_corrected_data.dropna(subset=['reference.cotton'])
    # Put all measurements of certain specimen into test data set
    test_data = data_clean.loc[data_clean['reference.specimen'] == 1]
    training_data = data_clean[~data_clean.isin(test_data)].dropna()  
    # Prepare the feature set (exclude non-spectral columns)
    X_test = test_data.drop(columns=['Unnamed: 0', 'reference.pet', 'reference.cotton', 'reference.specimen', 'reference.area', 'reference.spot', 'reference.measuring_date'])
    X_test.columns = X_test.columns.str.replace('spectra.', '')
    X_training = training_data.drop(columns=['Unnamed: 0', 'reference.pet', 'reference.cotton', 'reference.specimen', 'reference.area', 'reference.spot', 'reference.measuring_date'])
    X_training.columns = X_training.columns.str.replace('spectra.', '')
    # Prepare the target column (cotton content)
    y_test = test_data['reference.cotton']
    y_training = training_data['reference.cotton']
    return (X_training, X_test, y_training, y_test)


models = {
    "SVR": RandomizedSearchCV(
        SVR(),
        param_distributions={
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [10, 1, 0.1, 0.01, 0.001],
            'gamma': np.logspace(-2, 2, 5),
            'epsilon': [0.01, 0.1, 0.5, 1.0],
            'degree': [2, 3, 4, 5],
        },
        n_iter=10, random_state=42
    ),
    "Kernel Ridge": RandomizedSearchCV(
        KernelRidge(kernel="rbf"),
        param_distributions={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
        n_iter=10, random_state=42
    ),
    "Random Forest Regressor": RandomizedSearchCV(
        RandomForestRegressor(),
        param_distributions={
            "n_estimators": [10, 20, 50, 100, 200, 500],
            "max_depth": [3, 5, 10, 20, 50, 100, None],
            "min_samples_split": [2,5,10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        },
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
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "best_params": model.best_params_,
        "training_time": training_time,
        "prediction_time": prediction_time,
        "rmse": rmse,
        "r2": r2
    }