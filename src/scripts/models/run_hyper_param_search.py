import os
os.chdir("../../..")
# Make sure working directory is root folder
print(os.getcwd())

import sys
# Needed for Python to find the util modules
sys.path.insert(0, "src")

import json
import util.m00_general_util as util
import util.m06_regression_models as model_util
import util.m06_model_prep as prep_util
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


input_dir = "temp/fixed_cotton/input"
models = ["CNN"]
output_file = "temp/fixed_cotton/model_output_test_kernel.json"
plot_path = "temp/fixed_cotton/plots"

specimen_split = {"type": "attribute", "attribute": "reference.specimen", "test_value": 1 }
random_split = {"type": "random", "test_size": 0.25 }

settings = {
    "split": specimen_split,
    "scale":  True,
    "pca":  {"enabled": False, "n_comps": 15}
}

while os.path.exists(output_file):
    output_file = output_file + "1"
    
print(f"Results will be saved to {output_file}")

def create_pipeline_steps_from_settings(settings):
    pipeline_steps = []

    if(settings["scale"]):
        pipeline_steps.append(("scaler", StandardScaler()))

    pca_setting = settings["pca"]
    if(pca_setting["enabled"]):
        pipeline_steps.append(("pca", PCA(n_components=pca_setting["n_comps"])))

    return pipeline_steps

csv_files = util.get_files(input_dir)
output = []
for csv_file in csv_files:
    csv_path = os.path.join(input_dir, csv_file)
    baseline_corr_type = prep_util.get_baseline_corr_type(csv_file)
    data = prep_util.load_feature_set_from_csv(csv_path)
    # Data set split
    train_test_split = settings["split"]
    if(train_test_split["type"] == "attribute"):
        X_train, X_test, y_train, y_test, groups_train = prep_util.split_feature_set_with_attribute(data, train_test_split["attribute"], train_test_split["test_value"])
    elif(train_test_split["type"] == "random"):
        X_train, X_test, y_train, y_test, groups_train = prep_util.split_feature_set_randomly(data, train_test_split["test_size"])

    print(f"Train data set size: {len(X_train)}, test data set size: {len(X_test)}")

    #dist = model_util.median_squared_pairwise_distance(X_train)
    #print(f"{baseline_corr_type} has dist: {dist}, recommended gamma for RBF: {1/dist}")

    for model in models:
        model = model_util.get_model(model)
        pipeline_steps = create_pipeline_steps_from_settings(settings)
        pipeline_steps.append(("model", model.sk_model))
        pipeline = Pipeline(pipeline_steps)

        print(pipeline)
        model_output = model_util.hyper_param_search(
            pipeline, model, baseline_corr_type, X_train, X_test, y_train, y_test, plot_path, groups_train
        )
        result = {}
        result.update(
            model = model.name,
            baseline_corr = baseline_corr_type,
            Test_RMSE = model_output["test_rmse"],
            Test_R2 = model_output["test_r2"],
            Train_RMSE = model_output["train_rmse"],
            Train_R2 = model_output["train_r2"],
            CV_R2 = model_output["cv_r2"],
            training_time = model_output["training_time"],
            prediction_time = model_output["prediction_time"],
            best_params = model_output["best_params"],
            settings = settings
        )
        print(result)
        output.append(result)
        with open(output_file, "w") as outfile:
            json.dump(output, outfile, indent=4)
            print(f"Evaluating model {model} for {baseline_corr_type}")