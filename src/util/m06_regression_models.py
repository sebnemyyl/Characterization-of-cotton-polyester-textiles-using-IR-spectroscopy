from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor  # Import MLPRegressor for neural network
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import os

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

def run_pca(X_train, X_test, n_comps=50):
    pca = PCA(n_components=n_comps)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return (X_train_pca, X_test_pca)


models = {
    "SVR": RandomizedSearchCV(
        SVR(cache_size=7000),
        param_distributions={
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [10, 1, 0.1, 0.01, 0.001],
            'gamma': np.logspace(-2, 2, 5),
            'epsilon': [0.01, 0.1, 0.5, 1.0],
            'degree': [2, 3, 4, 5],
        },
        n_iter=10,
        cv=KFold(5, shuffle = True)
    ),
    "Kernel Ridge": RandomizedSearchCV(
        KernelRidge(kernel="rbf"),
        scoring="r2",
        param_distributions={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
        n_iter=10,
        cv=KFold(5, shuffle = True)
    ),
    "Random Forest": RandomizedSearchCV(
        RandomForestRegressor(),
        param_distributions={
            "n_estimators": [10, 20, 50, 100, 200, 500],
            "max_depth": [3, 5, 10, 20, 50, 100, None],
            "min_samples_split": [2,5,10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        },
        n_iter=10,
        cv=KFold(5, shuffle = True)
    ),
    "XGBoost": RandomizedSearchCV(
         XGBRegressor(),
         param_distributions={
             "n_estimators": [100, 200, 300],
             "max_depth": [3, 5, 7],
             "learning_rate": [0.01, 0.1, 0.2],
             "subsample": [0.8, 0.9, 1.0]
         },
         n_iter=10,
         cv=KFold(5, shuffle = True)
    ),
    "MLP": RandomizedSearchCV(
        MLPRegressor(max_iter=1000),
        param_distributions={
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'activation': ['relu', 'tanh']
        },
        n_iter=10,
        cv=KFold(5, shuffle = True)
    ),
    "PLS": RandomizedSearchCV(
        PLSRegression(),
        param_distributions={
            'n_components': range(5, 200, 5)
        },
        n_iter=10,  
        cv=KFold(5, shuffle = True)
    ),
    "KNN": RandomizedSearchCV(
        KNeighborsRegressor(),
        param_distributions={
            'n_neighbors': [3, 5, 10, 15, 20],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]  # For 'minkowski' metric, where p=1 is Manhattan and p=2 is Euclidean
        },
        n_iter=10,
        cv=KFold(5, shuffle=True)
    ),
}


def evaluate_model(model_name, baseline_corr, X_train, X_test, y_train, y_test, plot_path = ""):
    model = models[model_name]
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    cv_res = model.cv_results_
    cv_results = pd.DataFrame(model.cv_results_)[['mean_test_score', 'std_test_score', 'rank_test_score']]
    print(cv_results)

    # Predict
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # Calculate metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Training performance
    y_train_pred = model.predict(X_train)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    if plot_path != "":
        file_name = f"{model_name}_{baseline_corr}.png"
        file_path = os.path.join(plot_path, file_name)
        title = f"{model_name} for {baseline_corr}"
        create_prediction_plot(y_test, y_pred, y_train, y_train_pred, title)
        plt.savefig(file_path)
        print(f"Saving Plot to {file_path}")

    return {
        "model": model,
        "best_params": model.best_params_,
        "training_time": training_time,
        "prediction_time": prediction_time,
        "test_rmse": rmse,
        "test_r2": r2,
        "train_rmse": train_rmse,
        "train_r2": train_r2
    }

def create_prediction_plot(y_test, y_pred, y_train, y_train_pred, title = "Prediction plot"):
    test_pred = plt.scatter(y_test, y_pred, marker="x", c='b')
    train_pred = plt.scatter(y_train, y_train_pred, marker="o", facecolors='none', edgecolors='g')
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'k-')
    plt.legend((train_pred, test_pred), ("Train", "Test"),loc = "lower right")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)