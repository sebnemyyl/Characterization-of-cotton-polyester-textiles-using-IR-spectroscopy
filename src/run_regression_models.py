from sklearn.model_selection import train_test_split
import os
import util.m06_regression_models as model_util


model_names = ["SVR", "Kernel Ridge", "MLP"]


# Load the dataset
print(os.getcwd())
my_file = "temp/nir/spectra_nir_als.csv"
(X, y) = model_util.load_feature_set(my_file)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run the model
my_name = "SVR"
results = model_util.hyper_param_search(my_name, X_train, X_test, y_train, y_test)
print(f"{my_name} done. Best params: {results['best_params']}")



#Train and evaluate all models
# results = {}
# for name, model in models.items():
    # print(f"Training {name}...")
    # results[name] = model_util.evaluate_model(model, X_train, X_test, y_train, y_test)
    # print(f"{name} done. Best params: {results[name]['best_params']}")
# 
#Display results
# for name, res in results.items():
    # print(f"\n{name} Results:")
    # print(f"  - R2 Score: {res['r2']:.3f}")
    # print(f"  - MSE: {res['mse']:.3f}")
    # print(f"  - Training Time: {res['training_time']:.3f}s")
    # print(f"  - Prediction Time: {res['prediction_time']:.3f}s")



#
# # Cross-validate SVR
# t0 = time.time()
# svr_cv_scores = cross_val_score(svr, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
# svr_fit = time.time() - t0
# print(f"SVR cross-validation MSE: {-svr_cv_scores.mean():.3f} (+/- {svr_cv_scores.std():.3f})")
# print(f"SVR fitted in %.3f s" % svr_fit)
#
# # Cross-validate Kernel Ridge Regression
# t0 = time.time()
# kr_cv_scores = cross_val_score(kr, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
# kr_fit = time.time() - t0
# print(f"KRR cross-validation MSE: {-kr_cv_scores.mean():.3f} (+/- {kr_cv_scores.std():.3f})")
# print(f"KRR fitted in %.3f s" % kr_fit)


