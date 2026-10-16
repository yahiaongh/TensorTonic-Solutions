import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    N = len(y_pred)
    mse = np.sum((y_pred - y_true) * (y_pred - y_true)) / N
    return mse