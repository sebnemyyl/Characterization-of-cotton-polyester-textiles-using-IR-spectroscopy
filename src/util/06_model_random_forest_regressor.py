import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
from sklearn.model_selection import GridSearchCV

# Load the dataset
print(os.getcwd())
my_path = "../../temp"
file = "spectra_treated_snv_nir.csv"
baseline_corrected_data = pd.read_csv(f'{my_path}/{file}', sep=',', header=0)

# Convert the cotton column to numeric and handle errors
baseline_corrected_data['reference.cotton'] = pd.to_numeric(baseline_corrected_data['reference.cotton'], errors='coerce')

# Drop rows with missing cotton values
data_clean = baseline_corrected_data.dropna(subset=['reference.cotton'])

# Prepare the feature set (exclude non-spectral and cotton columns)
X = data_clean.drop(columns=['Unnamed: 0', 'reference.pet', 'reference.cotton', 'reference.specimen', 'reference.area', 'reference.spot', 'reference.measuring_date'])

X.columns = X.columns.str.replace('spectra.', '')

# Convert all features to numeric (replace commas in numbers with dots)
#X = X.apply(lambda x: pd.to_numeric(x.str.replace(',', '.'), errors='coerce'))

# Prepare the target column (cotton content)
y = data_clean['reference.cotton']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

parameters = {
    'n_estimators': [100, 150],
    'max_depth': [3,4],
}

rf_regressor = RandomForestRegressor(random_state=42)

reg = GridSearchCV(rf_regressor, parameters)
reg.fit(X_train, y_train)

# Initialize the Random Forest Regressor
#rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
#rf_regressor.fit(X_train, y_train)

# Predict the cotton content on the test set
y_pred = reg.predict(X_test)

# Evaluate the model by calculating the mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


results = pd.DataFrame(zip(y_test, y_pred, y_test - y_pred), columns = ['y_test', 'y_pred', 'error'])
results.head(10)