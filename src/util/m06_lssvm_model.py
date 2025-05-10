import numpy as np
from scipy.linalg import solve
from sklearn.base import BaseEstimator, RegressorMixin


# Kernel
def polynomial_kernel(X1, X2, degree=3, coef0=1, scale=1):
    return (scale * np.dot(X1, X2.T) + coef0) ** degree



# LS-SVM wrapped as sklearn estimator
class LSSVMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=1.0, sigma=1.0):
        self.gamma = gamma
        self.sigma = sigma

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        n_samples = X.shape[0]
        K = polynomial_kernel(X, X, self.sigma)
        Omega = K + np.eye(n_samples) / self.gamma

        ones = np.ones((n_samples, 1))
        A = np.block([[0, ones.T], [ones, Omega]])
        b_vec = np.concatenate(([0], y))

        sol = solve(A, b_vec)
        self.b_ = sol[0]
        self.alpha_ = sol[1:]
        return self

    def predict(self, X):
        K_test = polynomial_kernel(X, self.X_train_, self.sigma)
        return np.dot(K_test, self.alpha_) + self.b_

    def get_params(self, deep=True):
        return {"gamma": self.gamma, "sigma": self.sigma}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


# Factory function to create the model (for your model list)
def lssvm_regressor():
    return LSSVMRegressor()
