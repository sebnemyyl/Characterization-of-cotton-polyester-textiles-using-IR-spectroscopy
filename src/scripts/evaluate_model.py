import sys
# Needed for Python to find the util modules
sys.path.insert(0, "src")
sys.path.insert(0, "..")

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay
import os
import util.m06_regression_models as model_util
from sklearn.metrics import r2_score, root_mean_squared_error


print(os.getcwd())
csv_path = "../../temp/spectra_treated/nir/spectra_nir_regression_snv.csv"
model_name = "Kernel Ridge"
#output_file = "../../temp/spectra_treated/nir/kernel_model_output.json"

#X, y = model_util.load_feature_set(csv_path)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
X_train, X_test, y_train, y_test = model_util.split_feature_set_with_specimen(csv_path)
print(len(y_test))
print("Run model")
model = model_util.models[model_name]
# Train the model
model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)

plt.scatter(y_test, y_pred)
p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel("Actual")
plt.ylabel("Predicted")
#display = PredictionErrorDisplay(y_true=y_test, y_pred=y_pred)
#display.plot()
plt.show()