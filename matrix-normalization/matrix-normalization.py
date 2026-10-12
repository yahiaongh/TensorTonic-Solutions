import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    matrix = np.array(matrix, dtype=float)
    
    # Map the norm types. 
    # If axis is None, 'l2' must use None (Frobenius/Euclidean) instead of 2 (spectral).
    if axis is None and norm_type == "l2":
        ord_val = None
    else:
        d = {"l1": 1, "l2": 2, "max": np.inf}
        ord_val = d[norm_type]
        
    norm = np.linalg.norm(matrix, ord=ord_val, axis=axis, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    return matrix / norm
