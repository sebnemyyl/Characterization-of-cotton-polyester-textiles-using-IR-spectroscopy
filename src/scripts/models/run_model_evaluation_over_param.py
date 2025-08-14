import os
os.chdir("../../..")
print(os.getcwd())

import sys
# Needed for Python to find the util modules
sys.path.insert(0, "src")

import util.m00_general_util as util
import util.m06_regression_models as model_util
import util.m06_model_prep as prep_util
import numpy as np
import util.m06_cnn_model as cnn_model
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.kernel_ridge import KernelRidge
#from sklearn.cross_decomposition import PLSRegression
#from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression


# Settings
input_dir = "temp/fixed_cotton/input"
plot_path = "temp/fixed_cotton/plots"

# ## CNN
# model_kwargs = {
#    "optimizer": "adam",
#    "regularizer": 1e-2,
#    "kernel": 7
# }
#
# # CNN
# model = cnn_model.cnn_regressor
# param = "fit_kwargs"
# param_array = np.array([50])
# param_list = [{"epochs": val} for val in param_array]
# model.set_params(model_kwargs=model_kwargs)

# model = RandomForestRegressor(max_depth=20, min_samples_split=5 ,min_samples_leaf=4 , max_features="log2", bootstrap=False)
# param = "n_estimators"
# param_list = np.arange(5, 200, 5, dtype=int)

model = PLSRegression()
param = "n_components"
param_list = np.arange(1, 20, 1, dtype=int)

csv_files = util.get_files(input_dir)
output = []
for csv_file in csv_files:
    csv_path = os.path.join(input_dir, csv_file)
    baseline_corr_type = prep_util.get_baseline_corr_type(csv_file)
    data = prep_util.load_feature_set_from_csv(csv_path)
    X_train, X_test, y_train, y_test, groups_train = prep_util.split_feature_set_with_attribute(data, "reference.specimen", 1)
    #X_train, X_test = prep_util.run_pca(X_train, X_test, n_comps=15)

    #model_util.evaluate_cv_split(X_train, y_train, groups_train)
    model_util.evaluate_error_over_param(model, baseline_corr_type, param, param_list, X_train, X_test, y_train, y_test, plot_path, groups_train)