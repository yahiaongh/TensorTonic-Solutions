import math
import numpy as np
def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    p = y_pred
    max_p = np.where(p > eps, p, eps)
    print(max_p)
    _p = np.where(max_p > 1 - eps, 1 - eps, max_p)
    print(_p)
    LL = -(y_true * np.log(_p) + (1 - y_true) * np.log(1 - _p))
    return LL.tolist()