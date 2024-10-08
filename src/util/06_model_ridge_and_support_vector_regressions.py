from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time

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

# Prepare the target column (cotton content)
y = data_clean['reference.cotton']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svr = GridSearchCV(
    SVR(kernel="rbf", gamma=0.1),
    param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
)

kr = GridSearchCV(
    KernelRidge(kernel="rbf", gamma=0.1),
    param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
)


t0 = time.time()
svr.fit(X_train, y_train)
svr_fit = time.time() - t0
print(f"Best SVR with params: {svr.best_params_} and R2 score: {svr.best_score_:.3f}")
print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit)

t0 = time.time()
kr.fit(X_train, y_train)
kr_fit = time.time() - t0
print(f"Best KRR with params: {kr.best_params_} and R2 score: {kr.best_score_:.3f}")
print("KRR complexity and bandwidth selected and model fitted in %.3f s" % kr_fit)

sv_ratio = svr.best_estimator_.support_.shape[0] / X_train
#print("Support vector ratio: %.3f" % sv_ratio)
print("sv_ratio")
print(sv_ratio)

t0 = time.time()
y_svr = svr.predict(X_test)
svr_predict = time.time() - t0
print("SVR prediction in %.3f s" % svr_predict)

t0 = time.time()
y_kr = kr.predict(X_test)
kr_predict = time.time() - t0
print("KRR prediction in %.3f s" % kr_predict)

from sklearn.metrics import mean_squared_error

print(f"MSE SVR: {mean_squared_error(y_test, y_svr)}")
print(f"MSE KRR: {mean_squared_error(y_test, y_kr)}")