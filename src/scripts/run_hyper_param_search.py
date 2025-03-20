import sys
# Needed for Python to find the util modules
sys.path.insert(0, "src")
sys.path.insert(0, "..")


import os
import json
import util.m00_general_util as util
import util.m06_regression_models as model_util
import util.m06_model_prep as prep_util
import numpy as np


#os.chdir("../..")
print(os.getcwd())
input_dir = "temp/fixed_cotton/input"
models = ["Kernel Ridge poly"]
output_file = "temp/fixed_cotton/model_output_test_kernel.json"
plot_path = "temp/fixed_cotton/plots"

specimen_split = {"type": "attribute", "attribute": "reference.specimen", "test_value": 1 }
random_split = {"type": "random", "test_size": 0.25 }
train_test_split = specimen_split
pca_settings = {"enabled": True, "n_comps": 15}
#pca_settings = {"enabled": False}

while os.path.exists(output_file):
    output_file = output_file + "1"
    
print(f"Results will be saved to {output_file}")

csv_files = util.get_files(input_dir)
output = []
for csv_file in csv_files:
    csv_path = os.path.join(input_dir, csv_file)
    baseline_corr_type = prep_util.get_baseline_corr_type(csv_file)
    data = prep_util.load_feature_set_from_csv(csv_path)
    # Data set split
    if(train_test_split["type"] == "attribute"):
        X_train, X_test, y_train, y_test, groups_train = prep_util.split_feature_set_with_attribute(data, train_test_split["attribute"], train_test_split["test_value"])
    elif(train_test_split["type"] == "random"):
        X_train, X_test, y_train, y_test, groups_train = prep_util.split_feature_set_randomly(data, train_test_split["test_size"])

    print(f"Train data set size: {len(X_train)}, test data set size: {len(X_test)}")

    if (pca_settings["enabled"]):
        print("Run PCA")
        X_train, X_test = prep_util.run_pca(X_train, X_test, n_comps=pca_settings["n_comps"])

    #dist = model_util.median_squared_pairwise_distance(X_train)
    #print(f"{baseline_corr_type} has dist: {dist}, recommended gamma for RBF: {1/dist}")

    for model in models:
        model_output = model_util.hyper_param_search(
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
            CV_R2 = model_output["cv_r2"],
            training_time = model_output["training_time"],
            prediction_time = model_output["prediction_time"],
            best_params = model_output["best_params"],
            pca_settings = pca_settings,
            train_test_split = train_test_split
        )
        print(result)
        output.append(result)
        with open(output_file, "w") as outfile:
            json.dump(output, outfile, indent=4)
            print(f"Evaluating model {model} for {baseline_corr_type}")