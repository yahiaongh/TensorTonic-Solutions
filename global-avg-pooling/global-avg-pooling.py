import numpy as np

def global_avg_pool(x: list) -> np.ndarray:
    """
    Returns a spatially averaged NumPy array with shape (C,) or (N, C).
    """
    x = np.array(x)
    if len(x.shape) == 3:
        return gap(x)
    N, _, H, W = x.shape
    R = []
    for i in range(N):
        R.append(gap(x[i]))
    return np.array(R)
def gap(x):
    _, H, W = x.shape
    return np.sum(np.sum(x, axis=2),axis=1) / (H * W)