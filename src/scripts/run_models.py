import sys
# Needed for Python to find the util modules
sys.path.insert(0, "src")
sys.path.insert(0, "..")

from sklearn.model_selection import train_test_split

import os
import json
import util.m00_general_util as util
import util.m06_regression_models as model_util

def get_baseline_corr_type(csv_file):
    name_without_ending = os.path.splitext(csv_file)[0]
    parts = name_without_ending.split("_")
    return parts[-1]

print(os.getcwd())
input_dir = "../../temp/spectra_treated/nir"
models = [ "SVR"]
output_file = "../../temp/spectra_treated/nir/svr_model_output.json"


csv_files = util.get_csv_files(input_dir)
output = []
for csv_file in csv_files:
    csv_path = os.path.join(input_dir, csv_file)
    baseline_corr_type = get_baseline_corr_type(csv_file)
    #X, y = model_util.load_feature_set(csv_path)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_train, X_test, y_train, y_test = model_util.split_feature_set_with_specimen(csv_path)
    for model in models:
        print(f"Evaluating model {model} for {baseline_corr_type}")
        model_output = model_util.evaluate_model(model, X_train, X_test, y_train, y_test)
        result = {}
        result.update(
            model = model,
            baseline_corr = baseline_corr_type,
            Test_RMSE = model_output["test_rmse"],
            Test_R2 = model_output["test_r2"],
            Train_RMSE = model_output["train_rmse"],
            Train_R2 = model_output["train_r2"],
            training_time = model_output["training_time"],
            prediction_time = model_output["prediction_time"],
            best_params = model_output["best_params"],
        )
        print(result)
        output.append(result)

print(output)
with open(output_file, "w") as outfile:
    json.dump(output, outfile, indent=4)