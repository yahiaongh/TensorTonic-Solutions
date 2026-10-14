import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    x = np.array(x)
    n = x.shape[0]
    _x = np.mean(x)
    x = x - _x
    x = x.dot(x)
    s = np.sqrt(x / (n - 1))
    if s == 0:
        if _x == mu0:
            return 0
        return np.inf
    else:
        t = (_x - mu0) / (s / np.sqrt(n))
        return float(t)