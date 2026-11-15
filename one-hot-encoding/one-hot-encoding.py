import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    y = np.array(y)
    K = num_classes if num_classes else np.max(y) + 1
    N = y.size
    r = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            r[i][j] = 1 if j == y[i] else 0
    return r