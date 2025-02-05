import sys
# Needed for Python to find the util modules
sys.path.insert(0, "src")
sys.path.insert(0, "..")


import os
import json
import util.m00_general_util as util
import util.m06_regression_models as model_util

def get_baseline_corr_type(csv_file):
    name_without_ending = os.path.splitext(csv_file)[0]
    parts = name_without_ending.split("_")
    return parts[-1]

os.chdir("../..")
print(os.getcwd())
input_dir = "temp/spectra_treated/nir/balanced/corr"
models = ["Kernel Ridge"]
output_file = "temp/spectra_treated/nir/balanced/model_output_balanced.json"
plot_path = "temp/spectra_treated/nir/balanced/plots"


csv_files = util.get_files(input_dir)
output = []
for csv_file in csv_files:
    csv_path = os.path.join(input_dir, csv_file)
    baseline_corr_type = get_baseline_corr_type(csv_file)
    data = model_util.load_feature_set_from_csv(csv_path)
    #X_train, X_test, y_train, y_test, groups_train = model_util.split_feature_set_randomly(data)
    X_train, X_test, y_train, y_test, groups_train = model_util.split_feature_set_with_specimen(data)
    # Run PCA 
    #X_train, X_test = model_util.run_pca(X_train, X_test)
    for model in models:
        print(f"Evaluating model {model} for {baseline_corr_type}")
        model_output = model_util.evaluate_model(
            model, baseline_corr_type, X_train, X_test, y_train, y_test, plot_path, groups_train
            )
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
        with open(output_file, "w") as outfile:
            json.dump(output, outfile, indent=4)

print(output)
with open(output_file, "w") as outfile:
    json.dump(output, outfile, indent=4)