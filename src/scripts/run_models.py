import sys
# Needed for Python to find the util modules
sys.path.insert(0, "src")
sys.path.insert(0, "..")


import os
import json
import util.m00_general_util as util
import util.m06_regression_models as model_util
import util.m06_model_prep as prep_util


def get_baseline_corr_type(csv_file):
    name_without_ending = os.path.splitext(csv_file)[0]
    parts = name_without_ending.split("_")
    return parts[-1]

#os.chdir("../..")
print(os.getcwd())
input_dir = "temp/fixed_cotton/input_snv"
models = ["Kernel Ridge rbf"]
output_file = "temp/fixed_cotton/model_output_kernel_nothing.json"
plot_path = "temp/fixed_cotton/plots"

while os.path.exists(output_file):
    output_file = output_file + "1"
    
print(f"Results will be saved to {output_file}")

def eval_with_hyper_param_search(model, baseline_corr_type, X_train, X_test, y_train, y_test, plot_path, groups_train, output):
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
    )
    print(result)
    output.append(result)
    with open(output_file, "w") as outfile:
        json.dump(output, outfile, indent=4)

csv_files = util.get_files(input_dir)
output = []
for csv_file in csv_files:
    csv_path = os.path.join(input_dir, csv_file)
    baseline_corr_type = get_baseline_corr_type(csv_file)
    data = prep_util.load_feature_set_from_csv(csv_path)
    #X_train, X_test, y_train, y_test, groups_train = prep_util.split_feature_set_randomly(data)
    # Change method body to choose column for test data split
    X_train, X_test, y_train, y_test, groups_train = prep_util.split_feature_set_with_column(data)
    print(f"Train data set size: {len(X_train)}, test data set size: {len(X_test)}")
    # Run PCA 
    X_train, X_test = prep_util.run_pca(X_train, X_test, n_comps=15)
    #dist = model_util.median_squared_pairwise_distance(X_train)
    #print(f"{baseline_corr_type} has dist: {dist}, recommended gamma for RBF: {1/dist}")

    #model_util.evaluate_cv_split(X_train, y_train, groups_train)
    #model_util.evaluate_alpha( baseline_corr_type, X_train, X_test, y_train, y_test, plot_path, groups_train)
    for model in models:
         print(f"Evaluating model {model} for {baseline_corr_type}")
         eval_with_hyper_param_search(model, baseline_corr_type, X_train, X_test, y_train, y_test, plot_path, groups_train, output)
