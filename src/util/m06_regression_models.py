from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score, cross_validate
from sklearn.neural_network import MLPRegressor  # Import MLPRegressor for neural network
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os

import numpy as np

def load_feature_set_from_csv(csv_file):
    baseline_corrected_data = pd.read_csv(csv_file, sep=',', header=0)
    # Convert the cotton column to numeric and handle errors
    baseline_corrected_data['reference.cotton'] = pd.to_numeric(baseline_corrected_data['reference.cotton'], errors='coerce')
    # Drop rows with missing cotton values
    data_clean = baseline_corrected_data.dropna(subset=['reference.cotton'])
    return data_clean
 
# Prepare the feature set (exclude non-spectral columns) and
# remove spectra from column name
def get_X(data):
    X = data.loc[:,~data.columns.str.startswith('reference')]
    X = X.drop(columns=['Unnamed: 0'])
    X.columns = X.columns.str.replace('spectra.', '')
    return X

# Creates group list (combines cotton and specimen for unique specimen id)
def get_groups(data):
    groups = data['reference.cotton'] * 100000 + data['reference.specimen']
    unique_groups = np.unique(groups)
    print(f"Data set has {len(unique_groups)} number of unique specimen: {unique_groups}")
    return groups

def split_feature_set_randomly(data_clean):
    # Prepare the feature set (exclude non-spectral columns)
    # Prepare the target column (cotton content)
    y = data_clean['reference.cotton']
    X_train, X_test, y_train, y_test = train_test_split(data_clean, y, test_size=0.25)
    groups_train = get_groups(X_train)
    X_train = get_X(X_train)
    X_test = get_X(X_test)
    return (X_train, X_test, y_train, y_test, groups_train)

def split_feature_set_with_column(data_clean):
    # Put all measurements of certain column into test data set
    #test_data = data_clean.loc[data_clean['reference.batch'] == 2]
    test_data = data_clean.loc[data_clean['reference.specimen'] == 1]
    X_test = get_X(test_data)
    training_data = data_clean[~data_clean.isin(test_data)].dropna()  
    groups_train = get_groups(training_data)
    X_train = get_X(training_data)

    # Prepare the target column (cotton content)
    y_test = test_data['reference.cotton']
    y_train = training_data['reference.cotton']
    return (X_train, X_test, y_train, y_test, groups_train)

def run_pca(X_train, X_test, n_comps=50):
    pca = PCA(n_components=n_comps)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return (X_train_pca, X_test_pca)

default_n_iter = 40
default_cv = GroupKFold(n_splits=5)

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
        n_iter=default_n_iter,
        cv=default_cv
    ),
    "Kernel Ridge": RandomizedSearchCV(
        KernelRidge(kernel="rbf"),
        scoring="r2",
        param_distributions={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
        n_iter=default_n_iter,
        cv=default_cv
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
        n_iter=default_n_iter,
        cv=default_cv
    ),
    "XGBoost": RandomizedSearchCV(
         XGBRegressor(),
         param_distributions={
             "n_estimators": [100, 200, 300],
             "max_depth": [3, 5, 7],
             "learning_rate": [0.01, 0.1, 0.2],
             "subsample": [0.8, 0.9, 1.0]
         },
         n_iter=default_n_iter,
         cv=default_cv
    ),
    "MLP": RandomizedSearchCV(
        MLPRegressor(max_iter=1000),
        param_distributions={
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'activation': ['relu', 'tanh']
        },
        n_iter=default_n_iter,
        cv=default_cv
    ),
    "PLS": RandomizedSearchCV(
        PLSRegression(),
        param_distributions={
            'n_components': range(5, 200, 5)
        },
        n_iter=default_n_iter,
        cv=default_cv
    ),
    "KNN": RandomizedSearchCV(
        KNeighborsRegressor(),
        param_distributions={
            'n_neighbors': range(2, 50, 2),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        n_iter=default_n_iter,
        cv=default_cv
    ),
    "KNN Minkowski": RandomizedSearchCV(
        KNeighborsRegressor(metric='minkowski'),
        param_distributions={
            'n_neighbors': range(2, 50, 2),
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # p=1 is Manhattan and p=2 is Euclidean
        },
        n_iter=default_n_iter,
        cv=default_cv
    ),
}

#alpha_list =  [1e0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
alpha_list = np.geomspace(1e-10, 1.0, 30)

def evaluate_alpha(baseline_corr, X_train, X_test, y_train, y_test, plot_path = "", groups_train = None):
    model = KernelRidge(kernel="polynomial", degree=3, gamma=1/15)
    cv_rmse_values = []
    train_rmse_values = []
    test_rmse_values = []
    cv_r2_values = []
    train_r2_values = []
    test_r2_values = []
    print(f"Cross val for {baseline_corr}")
    for alpha in alpha_list:
        model.set_params(alpha=alpha)
        scoring_metrics = ['neg_root_mean_squared_error', 'r2']
        cross_val_res = cross_validate(model, X_train, y=y_train, scoring=scoring_metrics, groups=groups_train, cv=default_cv, return_estimator=True)
        cv_rmse_scores = cross_val_res['test_neg_root_mean_squared_error']
        cv_r2_scores = cross_val_res['test_r2']
        cv_rmse = np.mean(cv_rmse_scores) * -1
        cv_r2 = np.mean(cv_r2_scores) 
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
    file_name = f"alpha_rmse_{baseline_corr}.png"
    file_path = os.path.join(plot_path, file_name)
    create_plot(file_path, baseline_corr, train_rmse_values, cv_rmse_values, test_rmse_values)
    file_name = f"alpha_r2_{baseline_corr}.png"
    file_path = os.path.join(plot_path, file_name)
    create_plot(file_path, baseline_corr, train_r2_values, cv_r2_values, test_r2_values)

def create_plot(file_path, baseline_corr, train_values, cv_values, test_values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = alpha_list, y = train_values, mode="lines+markers", name="Train"))
    fig.add_trace(go.Scatter(x = alpha_list, y = cv_values, mode="lines+markers", name="CV"))
    fig.add_trace(go.Scatter(x = alpha_list, y = test_values, mode="lines+markers", name="Test"))
    fig.update_layout(xaxis_title=r"$\alpha$", xaxis_type="log", xaxis_tickformat="e", yaxis_title="Error")
    fig.update_layout(title = f"Plot for {baseline_corr}")
    #fig.show()
    fig.write_image(file_path)



def calc_error_metrics(actual, predicted):
    rmse = root_mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return (rmse, r2)


def hyper_param_search(model_name, baseline_corr, X_train, X_test, y_train, y_test, plot_path = "", groups_train = None):
    model = models[model_name]
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train, groups = groups_train)
    training_time = time.time() - start_time
    cv_res = model.cv_results_
    cv_results = pd.DataFrame(cv_res)[['mean_test_score', 'std_test_score', 'rank_test_score', 'params']]
    print(cv_results)
    print(model.best_estimator_)
    cv_r2 = model.best_score_
    print(cv_r2)

    # Predict
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    (test_rmse, test_r2) = calc_error_metrics(y_test, y_pred)

    # Training performance
    y_train_pred = model.predict(X_train)
    (train_rmse, train_r2) = calc_error_metrics(y_train, y_train_pred)

    if plot_path != "":
        file_name = f"{model_name}_{baseline_corr}.png"
        file_path = os.path.join(plot_path, file_name)
        title = f"{model_name} for {baseline_corr}"
        create_prediction_plot(y_test, y_pred, y_train, y_train_pred, title)
        plt.savefig(file_path)
        plt.clf()
        print(f"Saving Plot to {file_path}")

    return {
        "model": model,
        "best_params": model.best_params_,
        "training_time": training_time,
        "prediction_time": prediction_time,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
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


def evaluate_cv_split(X_train, y_train, groups_train):
    kFold = GroupKFold(n_splits=5)
    for i, (train_index, validation_index) in enumerate(kFold.split(X_train, groups=groups_train)):
        print(f"Fold {i}:")
        print(f"  Training size={len(train_index)}")
        print(f"  Validation size={len(validation_index)}")
        validation_values = y_train.iloc[validation_index]
        count = validation_values.value_counts().sort_index()
        count.plot(kind='bar', color='skyblue')
        print(count)
        plt.show()
        plt.clf()
