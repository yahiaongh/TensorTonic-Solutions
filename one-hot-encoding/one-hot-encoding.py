import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    K = num_classes or np.max(y) + 1
    N = len(y)
    R = np.zeros((N, K), dtype=float)
    for i in range(N):
        R[i][y[i]] = 1.0
    return R