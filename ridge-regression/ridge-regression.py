import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """
    X = np.array(X)
    y = np.array(y)
    lam_i = np.array([lam] * X.shape[1])
    return np.linalg.inv(X.T @ X + np.diag(lam_i)) @ X.T @ y