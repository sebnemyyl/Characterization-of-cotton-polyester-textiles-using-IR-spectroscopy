from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.layers import Input
from keras.wrappers import SKLearnRegressor
from itertools import product

import numpy as np


def mvn_augment(X, n_augmented=2):
    X_centered = X - X.mean(axis=0)
    cov_matrix = np.cov(X_centered.T)
    augmented = []
    for x in X:
        for _ in range(n_augmented):
            noise = np.random.multivariate_normal(mean=np.zeros(X.shape[1]), cov=cov_matrix)
            augmented.append(x + noise)
    X_aug = np.array(augmented)
    X_combined = np.concatenate([X, X_aug], axis=0)
    return X_combined


def add_snr_noise(signal, snr_db):
    # Add Gaussian noise to signal at specified SNR in dB.
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def create_cnn(X, y, optimizer, regularizer, kernel):
    # X_aug = mvn_augment(X, n_augmented=2)
    X_snr_aug = np.array([add_snr_noise(spectrum, snr_db=np.random.uniform(15, 30))
                          for spectrum in X])
    y_aug = np.tile(y, 3)

    features_normalized = X_snr_aug[:, :, np.newaxis]
    print(f"Using optimizer: {optimizer} with {len(X_snr_aug)} values")
    # dropout_rate = 0.3
    model = Sequential([
        Input(shape=(features_normalized.shape[1], 1)),

        Conv1D(filters=64, kernel_size=kernel, activation='relu', padding='same', kernel_regularizer=l2(regularizer)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=64, kernel_size=kernel, activation='relu', padding='same', kernel_regularizer=l2(regularizer)),
        BatchNormalization(),

        # Conv1D(filters=64, kernel_size=kernel, activation='relu', padding='same', kernel_regularizer=l2(regularizer)),
        # BatchNormalization(),

        Flatten(),

        Dense(128, activation='relu', kernel_regularizer=l2(regularizer)),
        Dropout(0.3),
        Dense(1)
    ])
    model.summary()
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
    "optimizer": ["adam"],
    "regularizer": [0.001],
    "kernel": [5]
}

fit_kwargs = {
    "epochs": [50],
    "batch_size": [8]
}

cnn_params = get_grid_from_dicts(
    prefix = "model__", 
    model_kwargs = model_kwargs, 
    fit_kwargs = fit_kwargs
)
