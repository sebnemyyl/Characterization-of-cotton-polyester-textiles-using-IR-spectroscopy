import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.regularizers import l2
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Input

# Load the data
input_dir = "../../temp/spectra_treated/nir/spectra_nir_regression_snv.csv"
data = pd.read_csv(input_dir)

# Extract target and features
target = data['reference.cotton'].values
features = data.filter(like='spectra').values

# Normalize the features
#scaler = MinMaxScaler()
#features_normalized = scaler.fit_transform(features)

# Reshape features for CNN input (samples, timesteps, features)
features_normalized = features[:, :, np.newaxis]

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features_normalized, target, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold = 1
r2_scores = []
rmse_scores = []

for train_idx, val_idx in kf.split(features_normalized):
    # Split data into training and validation sets
    X_train, X_val = features_normalized[train_idx], features_normalized[val_idx]
    y_train, y_val = target[train_idx], target[val_idx]

    # Build the model (re-initialize for each fold)
    model = Sequential([
        Input(shape=(features_normalized.shape[1], 1)),
        Conv1D(filters=32, kernel_size=7, activation='relu', input_shape=(features_normalized.shape[1], 1), kernel_regularizer=l2(0.01)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=7, activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(1)  # Regression output
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Reduce epochs to avoid long training times for each fold
        batch_size=16,
        verbose=1
    )

    # Evaluate on the validation set
    val_loss, val_rmse = model.evaluate(X_val, y_val, verbose=0)

    # Predict and calculate R² for the current fold
    y_val_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)

    # Store results
    r2_scores.append(r2)
    rmse_scores.append(val_rmse)

    print(f"Fold {fold}: R² = {r2:.4f}, RMSE = {val_rmse:.4f}")
    fold += 1


print(f"\nAverage R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")


# Save the trained model
#model.save('cnn_cotton_prediction_model.h5')


# import matplotlib.pyplot as plt
#
# # Plot training and validation loss
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Training and Validation Loss')
# plt.show()
#
#
#
# # Plot RMSE for training and validation
# plt.plot(history.history['root_mean_squared_error'], label='Training RMSE')
# plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
# plt.xlabel('Epochs')
# plt.ylabel('RMSE')
# plt.legend()
# plt.title('Training and Validation RMSE')
# plt.show()
#
#
# import numpy as np
#
#
# # Calculate R²
# r2 = r2_score(y_test, y_pred)
# print(f"R-squared: {r2}")
#
# # Scatter plot
# plt.scatter(y_test, y_pred, alpha=0.6)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title(f'Actual vs Predicted Values (R² = {r2:.2f})')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line
# plt.show()

import matplotlib.pyplot as plt

# Plot R² scores for each fold
plt.figure(figsize=(10, 5))
plt.bar(range(1, k + 1), r2_scores, color='skyblue', alpha=0.8)
plt.axhline(y=np.mean(r2_scores), color='red', linestyle='--', label=f"Average R² = {np.mean(r2_scores):.4f}")
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.title('R² Scores Across Folds')
plt.legend()
plt.show()

# Plot RMSE scores for each fold
plt.figure(figsize=(10, 5))
plt.bar(range(1, k + 1), rmse_scores, color='lightgreen', alpha=0.8)
plt.axhline(y=np.mean(rmse_scores), color='red', linestyle='--', label=f"Average RMSE = {np.mean(rmse_scores):.4f}")
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('RMSE Scores Across Folds')
plt.legend()
plt.show()
