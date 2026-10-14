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
    
def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    X = np.array(X)
    sigma = covariance_matrix(X)
    
    # Extract sample standard deviations from the diagonal of sigma (ensures ddof=1 consistency)
    std = np.sqrt(np.diag(sigma))
    
    # Compute the denominator matrix using the outer product of standard deviations
    denominator = np.outer(std, std)
    
    # Suppress warnings during division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        R = sigma / denominator
        
    # If any feature has zero variance, set its entire row and column to NaN (including diagonal)
    zero_var = (std == 0)
    if np.any(zero_var):
        R[zero_var, :] = np.nan
        R[:, zero_var] = np.nan
        
    return R