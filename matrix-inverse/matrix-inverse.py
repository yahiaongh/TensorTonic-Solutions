import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    try:
        return np.linalg.inv(A)
    except Exception:
        return None