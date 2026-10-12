import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    X = np.array(X)
    N = X.shape[0]
    mue = np.mean(X, axis=0)
    X_c = X - mue
    sigma = (X_c.T @ X_c) / (N - 1)
    return sigma