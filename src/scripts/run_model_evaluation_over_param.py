import sys
# Needed for Python to find the util modules
sys.path.insert(0, "src")
sys.path.insert(0, "..")


import os
import util.m00_general_util as util
import util.m06_regression_models as model_util
import util.m06_model_prep as prep_util
import numpy as np

from sklearn.kernel_ridge import KernelRidge

#os.chdir("../..")
print(os.getcwd())

# Settings
input_dir = "temp/fixed_cotton/input"
plot_path = "temp/fixed_cotton/plots"
model = KernelRidge(kernel="polynomial", degree=3)
param = "alpha"
param_list = np.geomspace(1e-10, 1.0, 30)


csv_files = util.get_files(input_dir)
output = []
for csv_file in csv_files:
    csv_path = os.path.join(input_dir, csv_file)
    baseline_corr_type = prep_util.get_baseline_corr_type(csv_file)
    data = prep_util.load_feature_set_from_csv(csv_path)
    X_train, X_test, y_train, y_test, groups_train = prep_util.split_feature_set_with_attribute(data, "reference.specimen", 1)
    X_train, X_test = prep_util.run_pca(X_train, X_test, n_comps=15)

    #model_util.evaluate_cv_split(X_train, y_train, groups_train)
    model_util.evaluate_error_over_param(model, baseline_corr_type, param, param_list, X_train, X_test, y_train, y_test, plot_path, groups_train)