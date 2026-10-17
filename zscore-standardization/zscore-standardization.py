import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    If a slice has a standard deviation at most eps, returns zeros for that slice.
    """
    X = np.array(X, dtype=float)
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    standardized = np.where(std <= eps, 0.0, (X - mean) / std)
    
    return standardized

