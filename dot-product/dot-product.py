import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    return np.array(x, dtype=float).dot(np.array(y, dtype=float)).item()