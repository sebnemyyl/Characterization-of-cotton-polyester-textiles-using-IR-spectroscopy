from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.layers import Input
from keras.wrappers import SKLearnRegressor
from itertools import product

import numpy as np

def create_cnn(X, y, optimizer):
    features_normalized = X[:, :, np.newaxis]
    print(f"Using optimizer: {optimizer} with {len(X)} values")
    dropout_rate = 0.3
    model = Sequential([
        Input(shape=(features_normalized.shape[1], 1)),
        Conv1D(filters=32, kernel_size=7, activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=7, activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(dropout_rate),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(dropout_rate),
        Dense(1)  # Regression output
    ])
    model.compile(optimizer=optimizer, loss='mse', metrics=[RootMeanSquaredError()])
    return model

cnn_regressor = SKLearnRegressor(
    model=create_cnn 
)

# Copied from https://github.com/keras-team/keras/pull/20599#issuecomment-2698338920
# Needs to be used like this because keras.wrappers don't fully support CV Search yet.
def get_grid_from_dicts(prefix="", model_kwargs=None, fit_kwargs=None):
    """This returns a param grid understood by GridSearchCV.

    We need to iterate over all keys of the two dicts, and for each of them,
    create something understood by the model object.

    For instance, if `model_kwargs` is `{'optimizer': ['adam', 'sgd']}`, and
    `fit_kwargs` is `{'epochs': [10, 20, 50]}`, this function will return
    [
        {'model_kwargs': [{'optimizer': 'adam'}], 'fit_kwargs': [{'epochs': 10}]},
        {'model_kwargs': [{'optimizer': 'adam'}], 'fit_kwargs': [{'epochs': 20}]},
        {'model_kwargs': [{'optimizer': 'adam'}], 'fit_kwargs': [{'epochs': 50}]},
        {'model_kwargs': [{'optimizer': 'sgd'}], 'fit_kwargs': [{'epochs': 10}]},
        {'model_kwargs': [{'optimizer': 'sgd'}], 'fit_kwargs': [{'epochs': 20}]},
        {'model_kwargs': [{'optimizer': 'sgd'}], 'fit_kwargs': [{'epochs': 50}]},
    ]
    """

    # Get all possible combinations of model parameters
    model_keys = list(model_kwargs.keys())
    model_values = [model_kwargs[k] for k in model_keys]
    model_combinations = list(product(*model_values))

    # Get all possible combinations of fit parameters
    fit_keys = list(fit_kwargs.keys())
    fit_values = [fit_kwargs[k] for k in fit_keys]
    fit_combinations = list(product(*fit_values))

    param_grid = []
    # Generate all possible combinations of model and fit parameters
    for model_combo in model_combinations:
        model_dict = dict(zip(model_keys, model_combo))
        for fit_combo in fit_combinations:
            fit_dict = dict(zip(fit_keys, fit_combo))
            param_grid.append(
                {
                    f"{prefix}model_kwargs": [model_dict],
                    f"{prefix}fit_kwargs": [fit_dict],
                }
            )

    return param_grid

# Every param here needs to be defined in function create_cnn
model_kwargs = {
    "optimizer": ["adam", "rmsprop"]
}

fit_kwargs = {
    "epochs": [10, 20]
}

cnn_params = get_grid_from_dicts(
    prefix = "model__", 
    model_kwargs = model_kwargs, 
    fit_kwargs = fit_kwargs
)
