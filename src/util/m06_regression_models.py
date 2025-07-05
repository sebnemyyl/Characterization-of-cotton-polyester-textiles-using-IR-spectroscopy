import sys
# Needed for Python to find the util modules
sys.path.insert(0, "src")
sys.path.insert(0, "..")

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from scipy.spatial.distance import pdist
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os

import util.m06_model_prep as prep_util
import util.m06_cnn_model as cnn_util
import util.m06_lssvm_model as lssvm_util

default_n_iter = 40
default_cv = GroupKFold(n_splits=4)

class Model:
    def __init__(self, name, sk_model, param_distributions):
        self.name = name
        self.sk_model = sk_model
        self.param_distributions = param_distributions

model_list = [
    Model(
        name="SVR rbf", 
        sk_model=SVR(cache_size=7000, kernel="rbf"),
        param_distributions={
            'C': np.logspace(-3, 2, 6),
            'gamma': np.logspace(-2, 6, 9),
            'epsilon': np.logspace(-3, 1, 5)
        }
    ),
    Model(
        name="SVR poly",
        sk_model=SVR(cache_size=7000, kernel='poly'),
        param_distributions={
            'C': np.logspace(-3, 2, 6),
            'gamma': np.logspace(-2, 6, 9) ,
            'epsilon': np.logspace(-3, 1, 5),
            'degree': [2, 3, 4, 5]
        }
    ),
    Model(
        name="SVR sigmoid",
        sk_model=SVR(cache_size=7000, kernel="sigmoid"),
        param_distributions={
            'C': np.logspace(-3, 2, 6),
            'gamma': np.logspace(-2, 6, 9),
            'epsilon': np.logspace(-3, 1, 5)
        },
    ),
    Model(
            name="LSSVM",
            sk_model=lssvm_util.lssvm_regressor(),
            param_distributions={
                'sigma': np.logspace(-2, 2, 5),
                'gamma': np.logspace(-2, 2, 5)            },
    ),
    Model(
        name="Kernel Ridge rbf",
        sk_model=KernelRidge(kernel="rbf"),
        param_distributions={
            "alpha": np.logspace(-3, 2, 6),
            "gamma": np.logspace(-2, 6, 9) # Gamma should be close to median squared pairwise distance
        }
    ),
    Model(
        name="Kernel Ridge poly",
        sk_model=KernelRidge(kernel="poly"),
        param_distributions={
            "alpha": np.logspace(-3, 2, 10),
            'degree': [2, 3, 4, 5]
        }
    ),
    Model(
        name="KNN",
        sk_model=KNeighborsRegressor(),
        param_distributions={
            'n_neighbors': range(2, 50, 2),
            'metric': ['euclidean', 'manhattan']
        }
    ),
    Model(
        name="Random Forest",
        sk_model= RandomForestRegressor(),
        param_distributions={
            "n_estimators": [10, 20, 50, 100, 200, 500],
            "max_depth": [3, 5, 10, 20, 50, 100, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        }
    ),
    Model(
        name="XGBoost",
        sk_model= XGBRegressor(),
        param_distributions={
             "n_estimators": [100, 200, 300],
             "max_depth": [3, 5, 7],
             "learning_rate": [0.01, 0.1, 0.2],
             "subsample": [0.8, 0.9, 1.0]
         }
    ),
    Model(
        name= "MLP",
        sk_model=MLPRegressor(max_iter=1000),
        param_distributions={
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'activation': ['relu', 'tanh']
    },
),
    Model(
        name="PLS",
        sk_model= PLSRegression(),
        param_distributions={
            'n_components': range(5, 200, 5)
    },
),

    Model(
        name="KNN Minkowski",
        sk_model= KNeighborsRegressor(metric='minkowski'),
        param_distributions={
            'n_neighbors': range(2, 50, 2),
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # p=1 is Manhattan and p=2 is Euclidean
    },

),
    Model(
        name="CNN",
        sk_model=cnn_util.cnn_regressor,
        param_distributions=cnn_util.cnn_params
    )
]

def get_model(name):
    return next((model for model in model_list if model.name == name), None)

def combine_cv_search_params(pca_params={}, model_params={}):
    return {f'pca__{key}': value for key, value in pca_params.items()} | \
           {f'model__{key}': value for key, value in model_params.items()}


def evaluate_error_over_param(model, baseline_corr, param_name, param_list, X_train, X_test, y_train, y_test, plot_path = "", groups_train = None):
    cv_rmse_values = []
    train_rmse_values = []
    test_rmse_values = []
    cv_r2_values = []
    train_r2_values = []
    test_r2_values = []
    print(f"Cross val for {baseline_corr}")
    cv = default_cv
    for param_value in param_list:
        model.set_params(**{param_name: param_value})
        scoring_metrics = ['neg_root_mean_squared_error', 'r2']
        cross_val_res = cross_validate(model, X_train, y=y_train, scoring=scoring_metrics, groups=groups_train, cv=cv, return_estimator=True)

        cv_r2_scores = cross_val_res['test_r2']
        cv_r2 = np.mean(cv_r2_scores) 

        cv_rmse_scores = cross_val_res['test_neg_root_mean_squared_error']
        rmse_std = np.std(cv_rmse_scores)
        cv_rmse = np.mean(cv_rmse_scores) * -1
        std_percentage = rmse_std / cv_rmse
        print(cv_rmse_scores)
        print(f"{param_name} {param_value} has RMSE Score: {cv_rmse} with std dev: {rmse_std} ({std_percentage}%)")

        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        (train_rmse, train_r2) = calc_error_metrics(y_train, train_pred)
        test_pred = model.predict(X_test)
        (test_rmse, test_r2) = calc_error_metrics(y_test, test_pred)
        cv_rmse_values.append(cv_rmse)
        train_rmse_values.append(train_rmse)
        test_rmse_values.append(test_rmse)
        cv_r2_values.append(cv_r2)
        train_r2_values.append(train_r2)
        test_r2_values.append(test_r2)
    model_name = type(model).__name__
    create_comparison_plot(plot_path, model_name, baseline_corr, param_name, "rmse", param_list, train_rmse_values, cv_rmse_values, test_rmse_values)
    create_comparison_plot(plot_path, model_name, baseline_corr, param_name, "r2", param_list, train_r2_values, cv_r2_values, test_r2_values)

def create_comparison_plot(plot_path, model_name, baseline_corr, param_name, error_name, param_list, train_values, cv_values, test_values):
    fig = go.Figure()
    #param_name = 'dropout_rate' # change param name for CNN
    #param_list = [p[param_name] for p in param_list] # map param name for CNN
    fig.add_trace(go.Scatter(x = [d for d in param_list], y = train_values, mode="lines+markers", name="Train"))
    fig.add_trace(go.Scatter(x = [d for d in param_list], y = cv_values, mode="lines+markers", name="CV"))
    #fig.add_trace(go.Scatter(x = param_list, y = test_values, mode="lines+markers", name="Test"))
    fig.update_layout(xaxis_title=param_name, yaxis_title=error_name.upper())
    #title = f"{model_name}, {baseline_corr} {error_name} over {param_name}"
    #fig.update_layout(title = title, plot_bgcolor='white')
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(#type="log",
                     #tickformat=".0e",
                     showgrid=True,
                     gridwidth=0.6,
                     gridcolor='black',
                     griddash='dash')
    fig.update_yaxes(showgrid=True,
                     gridwidth=0.6,
                     gridcolor='black',
                     griddash='dash')
    #fig.show()
    file_name = f"{model_name}_{param_name}_{error_name}_{baseline_corr}.png"
    file_path = os.path.join(plot_path, file_name)
    fig.write_image(file_path)


def calc_error_metrics(actual, predicted):
    rmse = root_mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return (rmse, r2)


def hyper_param_search(pipeline, model, baseline_corr, X_train, X_test, y_train, y_test, plot_path = "", groups_train = None):
    if(model.name == "CNN"):
        # CNN parameters are loaded directly, because prefix is already added there
        param_grid = model.param_distributions
    else:
        param_grid = combine_cv_search_params(model_params=model.param_distributions)
    print(param_grid)
    # Enable when you want to use the predefined group split
    #default_cv = prep_util.predefined_group_split(groups_train)
    cv_search = RandomizedSearchCV(pipeline, param_grid, cv=default_cv, n_iter=default_n_iter)
    # Train the model
    start_time = time.time()
    cv_search.fit(X_train, y_train, groups = groups_train)
    training_time = time.time() - start_time
    cv_res = cv_search.cv_results_
    cv_results = pd.DataFrame(cv_res)[['mean_test_score', 'std_test_score', 'rank_test_score', 'params']]
    print(cv_results)
    print(cv_search.best_estimator_)
    cv_r2 = cv_search.best_score_
    print(cv_r2)

    # Predict
    start_time = time.time()
    y_pred = cv_search.predict(X_test)
    prediction_time = time.time() - start_time

    (test_rmse, test_r2) = calc_error_metrics(y_test, y_pred)


    y_train_pred = cv_search.predict(X_train)
    (train_rmse, train_r2) = calc_error_metrics(y_train, y_train_pred)

    if plot_path != "":
        file_name = f"{model.name}_{baseline_corr}.png"
        file_path = os.path.join(plot_path, file_name)
        title = f"{model.name} for {baseline_corr}"
        create_prediction_plot(y_test, y_pred, y_train, y_train_pred, title)
        plt.savefig(file_path)
        plt.clf()
        if model.name == "PLS":
            create_and_save_pls_residual_plot(y_test, y_pred, plot_path, model, baseline_corr)
        print(f"Saving Plot to {file_path}")

    return {
        "model": model.name,
        "best_params": cv_search.best_params_,
        "training_time": training_time,
        "prediction_time": prediction_time,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "cv_r2": cv_r2
    }

def create_and_save_pls_residual_plot(y_test, y_pred, plot_path, model, baseline_corr):
    file_name = f"residual_{model.name}_{baseline_corr}.png"
    file_path = os.path.join(plot_path, file_name)
    #title = f"Residual {model.name} for {baseline_corr}"
    print("Saving residual plot")
    #y_pred = y_pred.ravel()  # Flatten to 1D
    residuals = y_test - y_pred
    sample_idx = np.arange(1, len(y_test) + 1)
    plt.figure(figsize=(8, 5))
    plt.scatter(sample_idx, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.grid(True)
    plt.savefig(file_path)
    plt.clf()


def create_prediction_plot(y_test, y_pred, y_train, y_train_pred, title="Prediction plot"):
    plt.figure(figsize=(8, 6))

    # Scatter plots
    train_pred = plt.scatter(y_train, y_train_pred, marker="o", facecolors='none', edgecolors='forestgreen',
                             label="Train", alpha=0.7)
    test_pred = plt.scatter(y_test, y_pred, marker="x", c='royalblue', label="Test", alpha=0.9)

    # 1:1 Line
    p1 = max(max(y_pred), max(y_test), max(y_train_pred), max(y_train))
    p2 = min(min(y_pred), min(y_test), min(y_train_pred), min(y_train))
    plt.plot([p1, p2], [p1, p2], 'k--', linewidth=1)

    # Labels and styling
    plt.xlabel("Actual", fontsize=12)
    plt.ylabel("Predicted", fontsize=12)
    #plt.title(title, fontsize=14)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.box(False)  # cleaner look


def evaluate_cv_split(X_train, y_train, groups_train):
    cv = default_cv
    for i, (train_index, validation_index) in enumerate(cv.split(X_train, groups=groups_train)):
        print(f"Fold {i}:")
        print(f"  Training size={len(train_index)}")
        print(f"  Validation size={len(validation_index)}")
        validation_values = y_train.iloc[validation_index]
        count = validation_values.value_counts().sort_index()
        count.plot(kind='bar', color='skyblue')
        print(count)
        plt.show()
        plt.clf()

def median_squared_pairwise_distance(X):
    # Compute pairwise Euclidean distances
    pairwise_distances = pdist(X, metric='euclidean')
    # Square the distances
    squared_distances = pairwise_distances ** 2
    # Return the median of the squared distances
    return np.median(squared_distances)