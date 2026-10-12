import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    x = np.array(x, dtype=float)
    p = float(p)
    pmf = np.where(x == 0, 1 - p, p)
    return {"pmf": pmf, "mean": p, "variance": p * (1- p)}