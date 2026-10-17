import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    N = len(y_true)
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=float)
    return -1 / N * np.sum(np.log([y_pred[i][y_true[i]] for i in range(N)]))