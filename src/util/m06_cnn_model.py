from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.layers import Input
from keras.wrappers import SKLearnRegressor



def create_cnn(X, y, optimizer):
    print(f"Using optimizer: {optimizer} with {len(X)} values")
    dropout_rate = 0.3
    model = Sequential([
        Input(shape=(X.shape[1], 1)),
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
    model=create_cnn, 
    model_kwargs={
        "optimizer": "rmsprop"
    },
    # fit_kwargs doesn't seem to work
    fit_kwargs={
        "epochs": 10
    }
)

# Param distribution needs to be adapted still
cnn_params = {}
#    'optimizer': 
#    'dropout_rate': [0.3, 0.5, 0.7],
#    'epochs': [10, 20],
#    'batch_size': [32, 64]