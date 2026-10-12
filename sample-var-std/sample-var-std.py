import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    mean = np.sum(x) / n
    t = x - mean 
    var = np.dot(t, t).item() / (n - 1)
    return {"variance": float(var), "standard_deviation": float(np.sqrt(var, dtype=float))}