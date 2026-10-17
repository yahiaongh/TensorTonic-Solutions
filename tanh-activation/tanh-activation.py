import numpy as np

def tanh(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.array(x)
    exp = np.exp(x)
    _exp = np.exp(-x)
    return (exp - _exp) / (exp + _exp)